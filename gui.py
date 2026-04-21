# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# gui.py v3.2 — BankBees User QTY Input Added
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import threading, time, json, os, csv, math
from datetime import datetime
from flask import Flask, jsonify, request, render_template_string
from config_loader  import load_config
from auth           import authenticate
from data_feed      import get_ltp, get_candles, get_bankbees_5min
from indicators     import calc_ema, calc_rsi
from strategy       import get_strategy, STRATEGIES, BankBeesMLScalper
from order_manager  import place_smart_order, check_exit_by_price
from logger         import log_trade

app   = Flask(__name__)
fyers = None
SYMBOLS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'symbols_state.json')
symbols_state = {}
symbols_lock  = threading.Lock()

# ── BankBees Global State ─────────────────────────────────────────────
bankbees_state = {
    "running"    : False,
    "in_trade"   : False,
    "last_check" : "—",
    "signal"     : "—",
    "prob"       : 0.0,
    "entry"      : 0.0,
    "target"     : 0.0,
    "sl"         : 0.0,
    "qty"        : 0,
    "user_qty"   : 10,      # ← User-defined qty (default 10)
    "exit_time"  : "—",
    "logs"       : [],
    "order_id"   : "—",
    "model_ok"   : False,
}
bankbees_lock   = threading.Lock()
bankbees_scalper_obj = None   # BankBeesMLScalper instance

def bb_log(msg, level="info"):
    t = datetime.now().strftime("%H:%M:%S")
    with bankbees_lock:
        bankbees_state["logs"].insert(0, {"t": t, "msg": msg, "lvl": level})
        bankbees_state["logs"] = bankbees_state["logs"][:40]

def bb_update(**kw):
    with bankbees_lock:
        bankbees_state.update(kw)

# ── BankBees Background Thread ────────────────────────────────────────
def bankbees_loop(stop_evt):
    global bankbees_scalper_obj
    bb_log("BankBees ML Scalper ਚਾਲੂ ✅", "success")

    while not stop_evt.is_set():
        if fyers is None:
            bb_log("Fyers not connected — waiting...", "warn")
            time.sleep(30)
            continue

        try:
            now_str = datetime.now().strftime("%H:%M:%S")
            bb_update(last_check=now_str)

            signal = bankbees_scalper_obj.run_bankbees(fyers)

            if signal:
                bb_update(
                    signal   = "BUY 🟢",
                    prob     = signal["prob"],
                    entry    = signal["entry_price"],
                    target   = signal["target"],
                    sl       = signal["sl"],
                    qty      = signal["qty"],
                    exit_time= str(signal["exit_time"]),
                )
                bb_log(
                    f"SIGNAL 🟢 entry={signal['entry_price']} "
                    f"sl={signal['sl']} tgt={signal['target']} "
                    f"prob={signal['prob']}% qty={signal['qty']}",
                    "success"
                )

                # Place order
                order_data = {
                    "symbol"      : signal["symbol"],
                    "qty"         : signal["qty"],
                    "type"        : 2,
                    "side"        : 1,
                    "productType" : "CNC",
                    "limitPrice"  : 0,
                    "stopPrice"   : 0,
                    "validity"    : "DAY",
                    "disclosedQty": 0,
                    "offlineOrder": False,
                }
                resp = fyers.place_order(data=order_data)
                if resp.get("s") == "ok":
                    oid = resp.get("id", "—")
                    bankbees_scalper_obj.mark_trade_open()
                    bb_update(in_trade=True, order_id=oid)
                    bb_log(f"ORDER ✅ id={oid}", "success")
                    _log_bb_trade(signal, oid)
                else:
                    bb_log(f"Order ਫੇਲ: {resp}", "error")
            else:
                bb_update(signal="HOLD")
                bb_log("ਕੋਈ signal ਨਹੀਂ", "info")

        except Exception as e:
            bb_log(f"Error: {e}", "error")

        # Sleep 5 min
        for _ in range(300):
            if stop_evt.is_set():
                break
            time.sleep(1)

    bb_update(running=False)
    bb_log("BankBees ਰੁਕਿਆ 🛑", "warn")

def _log_bb_trade(signal, order_id):
    row = {
        "order_id"   : order_id,
        "symbol"     : signal["symbol"],
        "signal"     : signal["signal"],
        "entry_price": signal["entry_price"],
        "target"     : signal["target"],
        "sl"         : signal["sl"],
        "qty"        : signal["qty"],
        "prob"       : signal["prob"],
        "time"       : str(signal["time"]),
        "exit_time"  : str(signal["exit_time"]),
    }
    file_exists = os.path.isfile("trades.csv")
    with open("trades.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# ─────────────────────────────────────────────────────────────────────

def safe_float(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return 0.0
    return v

def default_sym_state(symbol, cfg=None):
    c = cfg or {}
    return {
        "running": False, "price": 0.0, "ema_fast": 0.0, "ema_slow": 0.0,
        "rsi": 0.0, "signal": "HOLD", "position": None, "scan_no": 0,
        "logs": [], "last_tick": "", "error": "",
        "strategy":   c.get("strategy",      "ema_crossover"),
        "sl_pct":     c.get("stop_loss_pct",  1.5),
        "tgt_pct":    c.get("target_pct",     2.5),
        "lot_size":   c.get("lot_size",        1),
        "max_lots":   c.get("max_lots",        1),
        "ema_fast_p": c.get("ema_fast",        9),
        "ema_slow_p": c.get("ema_slow",       21),
        "entry_time": c.get("entry_time",  "09:20"),
        "exit_time":  c.get("exit_time",   "15:15"),
        "paper_trade":c.get("paper_trade",   True),
        "_stop_evt":  None, "_thread": None,
    }

def save_symbols():
    try:
        with symbols_lock:
            data = {}
            for sym, s in symbols_state.items():
                data[sym] = {k: v for k, v in s.items()
                             if not k.startswith('_') and k not in
                             ['running','price','ema_fast','ema_slow','rsi',
                              'signal','position','scan_no','logs','last_tick','error']}
        with open(SYMBOLS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Save error: {e}")

def load_symbols():
    try:
        if os.path.exists(SYMBOLS_FILE):
            with open(SYMBOLS_FILE) as f:
                data = json.load(f)
            for sym, cfg in data.items():
                symbols_state[sym] = default_sym_state(sym, cfg)
            print(f"  {len(data)} symbols loaded from file")
    except Exception as e:
        print(f"Load error: {e}")

def sym_log(symbol, msg, level="info"):
    t = datetime.now().strftime("%H:%M:%S")
    with symbols_lock:
        if symbol in symbols_state:
            symbols_state[symbol]["logs"].insert(0, {"t": t, "msg": msg, "lvl": level})
            symbols_state[symbol]["logs"] = symbols_state[symbol]["logs"][:40]

def sym_update(symbol, **kw):
    with symbols_lock:
        if symbol in symbols_state:
            symbols_state[symbol].update(kw)

def bot_loop(symbol, cfg, stop_evt):
    try:
        import pytz
        ist = pytz.timezone("Asia/Kolkata")
        def now_ist(): return datetime.now(ist).strftime("%H:%M")
    except:
        from datetime import timezone, timedelta
        tz = timezone(timedelta(hours=5, minutes=30))
        def now_ist(): return datetime.now(tz).strftime("%H:%M")

    strategy = get_strategy({"bot": cfg})
    position = None
    last_sig = 0
    sym_log(symbol, "Bot ਚਾਲੂ ✅ | " + cfg.get('strategy','ema_crossover'), "success")

    try:
        p = get_ltp(fyers, symbol)
        if p:
            sym_update(symbol, price=p, last_tick=datetime.now().strftime("%H:%M:%S"))
            sym_log(symbol, "Live: ₹{:,.2f}".format(p), "success")
    except Exception as e:
        sym_log(symbol, "Init error: " + str(e), "warn")

    while not stop_evt.is_set():
        hm = now_ist()
        if not (cfg.get("entry_time","09:20") <= hm <= cfg.get("exit_time","15:15")):
            try:
                p = get_ltp(fyers, symbol)
                if p: sym_update(symbol, price=p, last_tick=datetime.now().strftime("%H:%M:%S"))
            except: pass
            time.sleep(30)
            continue

        try:
            price = get_ltp(fyers, symbol)
            if not price or price == 0:
                time.sleep(5); continue
            sym_update(symbol, price=price, last_tick=datetime.now().strftime("%H:%M:%S"))

            if position:
                result = check_exit_by_price(position, price)
                sym_update(symbol, position=position)
                if result != "HOLD":
                    qty = position.get("filled_qty", cfg.get("lot_size",1))
                    pnl = round(((price-position["entry"]) if position["side"]=="BUY" else (position["entry"]-price)) * qty, 2)
                    log_trade(position["side"], position["entry"], price, result)
                    sym_log(symbol, "Position ਬੰਦ: {} | qty:{} | P&L: ₹{:+.2f}".format(result, qty, pnl),
                            "success" if pnl>=0 else "error")
                    close_sig = "SELL" if position["side"]=="BUY" else "BUY"
                    close_cfg = dict(cfg)
                    close_cfg["lot_size"] = qty
                    close_cfg["max_lots"] = 1
                    place_smart_order(fyers, close_sig, {"bot": close_cfg})
                    position = None
                    sym_update(symbol, position=None, signal="HOLD")
                time.sleep(5); continue

            if time.time() - last_sig >= 60:
                last_sig = time.time()
                df = get_candles(fyers, symbol, cfg.get("candle_timeframe","5"))
                if not df.empty:
                    ef  = calc_ema(df, cfg.get("ema_fast",9)).iloc[-1]
                    es  = calc_ema(df, cfg.get("ema_slow",21)).iloc[-1]
                    rsi = calc_rsi(df).iloc[-1]
                    sig = strategy.signal(df)
                    rsi_safe = float(rsi) if not math.isnan(float(rsi)) else 0.0
                    sno = symbols_state[symbol]["scan_no"] + 1
                    trend = "🟢" if ef > es else "🔴"
                    sym_update(symbol, ema_fast=round(float(ef),2), ema_slow=round(float(es),2),
                               rsi=rsi_safe, signal=sig, scan_no=sno)
                    sym_log(symbol, "Scan #{} | {} | ₹{:,.2f} | {} RSI:{:.1f}".format(sno, sig, price, trend, rsi_safe))

                    if sig in ("BUY","SELL"):
                        resp = place_smart_order(fyers, sig, {"bot": cfg})
                        if resp.get("ok"):
                            position = {
                                "side":       sig,
                                "entry":      resp.get("avg_price", price),
                                "filled_qty": resp.get("filled_qty", cfg.get("lot_size",1)),
                                "sl_price":   resp.get("sl_price", 0),
                                "tgt_price":  resp.get("tgt_price", 0),
                                "time":       datetime.now().strftime("%H:%M:%S")
                            }
                            sym_update(symbol, position=position)
                            sym_log(symbol, "✅ {} @ ₹{:,.2f} | SL:₹{:,.2f} | TGT:₹{:,.2f}".format(
                                sig, resp.get('avg_price',price), resp.get('sl_price',0), resp.get('tgt_price',0)), "success")
                        else:
                            sym_log(symbol, "Order: " + resp.get('msg',''), "warn")

        except Exception as e:
            sym_log(symbol, "Error: " + str(e), "error")
        time.sleep(5)

    sym_update(symbol, running=False, position=None, signal="HOLD")
    sym_log(symbol, "Bot ਰੁਕਿਆ 🛑", "warn")

def read_trades():
    if not os.path.exists("trades.csv"): return []
    with open("trades.csv") as f:
        return list(reversed(list(csv.DictReader(f))))[-50:]

import requests as _req
_sym_cache = {}
MARKET_URLS = {
    "NSE_FO":  "https://public.fyers.in/sym_details/NSE_FO_sym_master.json",
    "NSE_CM":  "https://public.fyers.in/sym_details/NSE_CM_sym_master.json",
    "MCX_COM": "https://public.fyers.in/sym_details/MCX_COM_sym_master.json",
    "BSE_CM":  "https://public.fyers.in/sym_details/BSE_CM_sym_master.json",
}

def sym_search(query, market="NSE_FO"):
    url = MARKET_URLS.get(market, MARKET_URLS["NSE_FO"])
    if market not in _sym_cache:
        try: _sym_cache[market] = _req.get(url, timeout=15).json()
        except: return []
    q, out = query.upper().strip(), []
    for ticker, info in _sym_cache[market].items():
        name = info.get("symDetails","") or info.get("symbolDesc","")
        if q in ticker.upper() or q in name.upper():
            out.append({"ticker": ticker, "name": name[:45],
                        "lot": info.get("minLotSize",1), "active": info.get("tradeStatus",0)})
    out.sort(key=lambda x: x["active"]==0)
    return out[:25]

# ── API Routes ────────────────────────────────────────────────────────

@app.route("/")
def index(): return render_template_string(HTML)

@app.route("/api/status")
def api_status():
    with symbols_lock:
        clean = {}
        for sym, s in symbols_state.items():
            row = {}
            for k, v in s.items():
                if k.startswith("_"): continue
                if isinstance(v, float): row[k] = safe_float(v)
                elif isinstance(v, (str, int, bool, list, dict, type(None))): row[k] = v
                else: row[k] = str(v)
            clean[sym] = row
    with bankbees_lock:
        bb = {}
        for k, v in bankbees_state.items():
            if k.startswith("_"): continue
            if isinstance(v, float): bb[k] = safe_float(v)
            elif isinstance(v, (str, int, bool, list, dict, type(None))): bb[k] = v
            else: bb[k] = str(v)
    return jsonify({"symbols": clean, "trades": read_trades(),
                    "strategies": list(STRATEGIES.keys()), "bankbees": bb})

@app.route("/api/bankbees/start", methods=["POST"])
def api_bb_start():
    global bankbees_scalper_obj
    if bankbees_state["running"]:
        return jsonify({"ok": True, "msg": "Already running"})
    try:
        config = load_config()
        bankbees_scalper_obj = BankBeesMLScalper(config)
        model_ok = bankbees_scalper_obj.scalper.model is not None

        # ── Saved user_qty restore ਕਰੋ ────────────────────────────
        saved_qty = bankbees_state.get("user_qty", 10)
        bankbees_scalper_obj.set_user_qty(saved_qty)

        stop_evt = threading.Event()
        t = threading.Thread(target=bankbees_loop, args=(stop_evt,), daemon=True)
        bb_update(running=True, model_ok=model_ok, _stop_evt=stop_evt, _thread=t)
        t.start()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})

@app.route("/api/bankbees/stop", methods=["POST"])
def api_bb_stop():
    with bankbees_lock:
        evt = bankbees_state.get("_stop_evt")
        if evt: evt.set()
        bankbees_state["running"] = False
    return jsonify({"ok": True})

@app.route("/api/bankbees/close_trade", methods=["POST"])
def api_bb_close():
    """Manually mark BankBees trade as closed"""
    global bankbees_scalper_obj
    if bankbees_scalper_obj:
        bankbees_scalper_obj.mark_trade_closed()
    bb_update(in_trade=False, signal="—", entry=0, target=0, sl=0, qty=0, order_id="—")
    bb_log("Trade manually closed ✅", "warn")
    return jsonify({"ok": True})

# ── ✅ NEW: User QTY Set Route ────────────────────────────────────────
@app.route("/api/bankbees/set_qty", methods=["POST"])
def api_bb_set_qty():
    """GUI ਤੋਂ user qty save + scalper ਨੂੰ apply ਕਰੋ"""
    global bankbees_scalper_obj
    data = request.json or {}
    try:
        qty = int(data.get("qty", 10))
        qty = max(qty, 1)   # minimum 1
    except (ValueError, TypeError):
        return jsonify({"ok": False, "msg": "ਸਹੀ number ਦਿਓ"})

    # State ਵਿੱਚ save ਕਰੋ (restart ਤੋਂ ਬਾਅਦ ਵੀ restore ਹੋਵੇ)
    bb_update(user_qty=qty)

    # ਜੇ scalper ਚੱਲ ਰਿਹਾ ਹੈ ਤਾਂ live update ਕਰੋ
    if bankbees_scalper_obj:
        bankbees_scalper_obj.set_user_qty(qty)

    bb_log(f"User QTY set → {qty} ✅", "success")
    return jsonify({"ok": True, "qty": qty})

@app.route("/api/add_symbol", methods=["POST"])
def api_add_symbol():
    data = request.json or {}
    symbol = data.get("symbol","").strip()
    if not symbol: return jsonify({"ok": False, "msg": "Symbol ਦੱਸੋ"})
    with symbols_lock:
        if symbol not in symbols_state:
            symbols_state[symbol] = default_sym_state(symbol, data)
    save_symbols()
    return jsonify({"ok": True})

@app.route("/api/remove_symbol", methods=["POST"])
def api_remove_symbol():
    symbol = (request.json or {}).get("symbol","")
    with symbols_lock:
        s = symbols_state.get(symbol)
        if s:
            if s.get("_stop_evt"): s["_stop_evt"].set()
            symbols_state.pop(symbol, None)
    save_symbols()
    return jsonify({"ok": True})

@app.route("/api/start_symbol", methods=["POST"])
def api_start_symbol():
    global fyers
    data = request.json or {}
    symbol = data.get("symbol","")
    with symbols_lock:
        s = symbols_state.get(symbol)
        if not s: return jsonify({"ok": False, "msg": "Symbol ਨਹੀਂ ਮਿਲਿਆ"})
        if s["running"]: return jsonify({"ok": True, "msg": "Already running"})
    try:
        config = load_config()
        if fyers is None: fyers = authenticate(config)
        bot_cfg = dict(config["bot"])
        with symbols_lock:
            s = symbols_state[symbol]
        bot_cfg.update({
            "index_symbol":  symbol,
            "strategy":      s["strategy"],
            "stop_loss_pct": s["sl_pct"],
            "target_pct":    s["tgt_pct"],
            "lot_size":      s["lot_size"],
            "max_lots":      s["max_lots"],
            "ema_fast":      s["ema_fast_p"],
            "ema_slow":      s["ema_slow_p"],
            "entry_time":    s["entry_time"],
            "exit_time":     s["exit_time"],
            "paper_trade":   s["paper_trade"],
        })
        stop_evt = threading.Event()
        t = threading.Thread(target=bot_loop, args=(symbol, bot_cfg, stop_evt), daemon=True)
        with symbols_lock:
            symbols_state[symbol]["_stop_evt"] = stop_evt
            symbols_state[symbol]["_thread"]   = t
            symbols_state[symbol]["running"]   = True
            symbols_state[symbol]["error"]     = ""
        t.start()
        return jsonify({"ok": True})
    except Exception as e:
        sym_update(symbol, running=False, error=str(e))
        return jsonify({"ok": False, "msg": str(e)})

@app.route("/api/stop_symbol", methods=["POST"])
def api_stop_symbol():
    symbol = (request.json or {}).get("symbol","")
    with symbols_lock:
        s = symbols_state.get(symbol)
        if s:
            if s.get("_stop_evt"): s["_stop_evt"].set()
            s["running"] = False
    return jsonify({"ok": True})

@app.route("/api/start_all", methods=["POST"])
def api_start_all():
    with symbols_lock:
        to_start = [sym for sym,s in symbols_state.items() if not s["running"]]
    started = 0
    for sym in to_start:
        with app.test_request_context('/api/start_symbol', method='POST',
            json={"symbol": sym}, content_type='application/json'):
            res = api_start_symbol()
            if res.get_json().get("ok"): started += 1
        time.sleep(0.5)
    return jsonify({"ok": True, "started": started})

@app.route("/api/stop_all", methods=["POST"])
def api_stop_all():
    stopped = []
    with symbols_lock:
        for sym, s in symbols_state.items():
            if s["running"]:
                if s.get("_stop_evt"): s["_stop_evt"].set()
                s["running"] = False
                stopped.append(sym)
    return jsonify({"ok": True, "stopped": stopped})

@app.route("/api/update_symbol", methods=["POST"])
def api_update_symbol():
    data = request.json or {}
    symbol = data.get("symbol","")
    with symbols_lock:
        s = symbols_state.get(symbol)
        if not s: return jsonify({"ok": False, "msg": "Not found"})
        s.update({
            "strategy":    data.get("strategy",    s["strategy"]),
            "sl_pct":      float(data.get("sl_pct",    s["sl_pct"])),
            "tgt_pct":     float(data.get("tgt_pct",   s["tgt_pct"])),
            "lot_size":    int(data.get("lot_size",    s["lot_size"])),
            "max_lots":    int(data.get("max_lots",    s["max_lots"])),
            "ema_fast_p":  int(data.get("ema_fast",    s["ema_fast_p"])),
            "ema_slow_p":  int(data.get("ema_slow",    s["ema_slow_p"])),
            "entry_time":  data.get("entry_time",  s["entry_time"]),
            "exit_time":   data.get("exit_time",   s["exit_time"]),
            "paper_trade": bool(data.get("paper_trade", s["paper_trade"])),
        })
    save_symbols()
    return jsonify({"ok": True})

@app.route("/api/search")
def api_search():
    q  = request.args.get("q","")
    mk = request.args.get("market","NSE_FO")
    if len(q) < 2: return jsonify([])
    return jsonify(sym_search(q, mk))

@app.route("/api/get_login_url")
def api_get_login_url():
    try:
        from fyers_apiv3 import fyersModel
        config = load_config()
        session = fyersModel.SessionModel(
            client_id=config["fyers"]["client_id"],
            secret_key=config["fyers"]["secret_key"],
            redirect_uri=config["fyers"]["redirect_uri"],
            response_type="code", grant_type="authorization_code")
        return jsonify({"ok": True, "url": session.generate_authcode()})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})

@app.route("/api/submit_token", methods=["POST"])
def api_submit_token():
    global fyers
    try:
        from fyers_apiv3 import fyersModel
        import datetime as dt
        data = request.json
        user_input = data.get("token","").strip()
        config = load_config()
        client_id    = config["fyers"]["client_id"]
        secret_key   = config["fyers"]["secret_key"]
        redirect_uri = config["fyers"]["redirect_uri"]
        token_file   = config["fyers"]["token_file"]
        text = user_input
        if "auth_code=" in text:   auth_code = text.split("auth_code=")[1].split("&")[0]
        elif "code=" in text:      auth_code = text.split("code=")[1].split("&")[0]
        else:                      auth_code = text
        session = fyersModel.SessionModel(client_id=client_id, secret_key=secret_key,
            redirect_uri=redirect_uri, response_type="code", grant_type="authorization_code")
        session.set_token(auth_code)
        response = session.generate_token()
        if response.get("s") != "ok":
            return jsonify({"ok": False, "msg": str(response)})
        access_token = response["access_token"]
        with open(token_file,"w") as f:
            json.dump({"access_token": access_token,
                       "date": dt.datetime.now().strftime("%Y-%m-%d")}, f, indent=4)
        fyers   = fyersModel.FyersModel(client_id=client_id, token=access_token, log_path="")
        profile = fyers.get_profile()
        name    = profile.get("data",{}).get("name","User") if profile.get("s")=="ok" else "User"
        return jsonify({"ok": True, "name": name})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})

HTML = r"""<!DOCTYPE html>
<html lang="pa">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Nifty Multi-Bot</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
<style>
body{background:#0f1117;color:#e0e0e0;font-family:'Segoe UI',sans-serif;font-size:13px;}
body.light{background:#f0f2f8;color:#1a1d27;}
body.light .card,body.light .sym-card,body.light .bb-card{background:#fff;border-color:#d0d4e0;}
body.light .sym-header,body.light .bb-header{background:#e8eaf0;border-color:#d0d4e0;}
body.light .header-bar{background:#fff;border-color:#d0d4e0;}
body.light input,body.light select{background:#f8f9fc;border-color:#c0c4d0;color:#1a1d27;}
body.light .log-box{background:#e8eaf0;}
body.light .search-results{background:#fff;}
body.light .search-item:hover{background:#e8eaf0;}
.card{background:#1a1d27;border:1px solid #2a2d3a;border-radius:10px;}
.card-header{background:#212435;border-bottom:1px solid #2a2d3a;font-weight:500;font-size:11px;text-transform:uppercase;color:#888;padding:8px 14px;}
input,select{background:#12151f;border:1px solid #2a2d3a;color:#e0e0e0;border-radius:6px;padding:5px 10px;font-size:12px;width:100%;}
input:focus,select:focus{outline:none;border-color:#2196f3;}
.btn-g{background:#1a3a1a;border:1px solid #4caf50;color:#4caf50;border-radius:6px;padding:5px 14px;font-size:12px;cursor:pointer;}
.btn-g:hover{background:#4caf50;color:#000;}
.btn-g:disabled{opacity:.4;cursor:not-allowed;}
.btn-r{background:#3a1a1a;border:1px solid #f44336;color:#f44336;border-radius:6px;padding:5px 14px;font-size:12px;cursor:pointer;}
.btn-r:hover{background:#f44336;color:#fff;}
.btn-b{background:#1a2a3a;border:1px solid #2196f3;color:#2196f3;border-radius:6px;padding:5px 14px;font-size:12px;cursor:pointer;}
.btn-b:hover{background:#2196f3;color:#fff;}
.btn-y{background:#3a2a00;border:1px solid #ffc107;color:#ffc107;border-radius:6px;padding:5px 14px;font-size:12px;cursor:pointer;}
.badge-run{background:#1a3a1a;color:#4caf50;border:1px solid #4caf50;border-radius:20px;padding:2px 8px;font-size:11px;}
.badge-stop{background:#3a1a1a;color:#f44336;border:1px solid #f44336;border-radius:20px;padding:2px 8px;font-size:11px;}
.badge-paper{background:#3a2a00;color:#ffc107;border:1px solid #ffc107;border-radius:20px;padding:2px 8px;font-size:11px;}
.badge-live{background:#1a3a1a;color:#4caf50;border:1px solid #4caf50;border-radius:20px;padding:2px 8px;font-size:11px;}
.badge-model-ok{background:#1a3a2a;color:#00e676;border:1px solid #00e676;border-radius:20px;padding:2px 8px;font-size:11px;}
.badge-model-no{background:#3a1a1a;color:#ff5252;border:1px solid #ff5252;border-radius:20px;padding:2px 8px;font-size:11px;}
.sym-card{background:#1a1d27;border:1px solid #2a2d3a;border-radius:10px;margin-bottom:12px;}
.sym-header{background:#212435;border-radius:10px 10px 0 0;padding:10px 14px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.sym-body{padding:12px 14px;}
/* BankBees Card */
.bb-card{background:#0d1a2a;border:1px solid #1a3a5a;border-radius:10px;margin-bottom:16px;}
.bb-header{background:#0f2035;border-radius:10px 10px 0 0;padding:10px 16px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;border-bottom:1px solid #1a3a5a;}
.bb-body{padding:14px 16px;}
.bb-metric{text-align:center;padding:8px 6px;background:#0a1520;border-radius:8px;border:1px solid #1a3a5a;}
.bb-metric-val{font-size:16px;font-weight:700;}
.bb-metric-lbl{font-size:10px;color:#4a7aaa;text-transform:uppercase;margin-top:2px;}
.bb-signal-buy{color:#00e676;font-size:18px;font-weight:800;}
.bb-signal-hold{color:#445566;font-size:18px;font-weight:700;}
.bb-prob-bar{height:6px;background:#1a3a5a;border-radius:3px;margin-top:4px;}
.bb-prob-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,#1565c0,#00e676);transition:width .5s;}
/* ✅ Qty input — BB header ਵਿੱਚ */
.bb-qty-wrap{display:flex;align-items:center;gap:6px;background:#071220;border:1px solid #1a5a8a;border-radius:8px;padding:4px 10px;}
.bb-qty-wrap label{font-size:10px;color:#4a7aaa;text-transform:uppercase;white-space:nowrap;margin:0;}
.bb-qty-input{background:transparent;border:none;color:#ffc107;font-size:14px;font-weight:700;width:60px;text-align:center;padding:0;}
.bb-qty-input:focus{outline:none;color:#ffeb3b;}
.bb-qty-set-btn{background:#1a3a5a;border:1px solid #2196f3;color:#2196f3;border-radius:5px;padding:2px 8px;font-size:11px;cursor:pointer;white-space:nowrap;}
.bb-qty-set-btn:hover{background:#2196f3;color:#fff;}
.metric{text-align:center;padding:6px 4px;}
.metric-val{font-size:17px;font-weight:600;}
.metric-lbl{font-size:10px;color:#666;text-transform:uppercase;}
.log-box{height:120px;overflow-y:auto;font-family:Consolas,monospace;font-size:11px;line-height:1.7;background:#0f1117;border-radius:6px;padding:8px;}
.bb-log-box{height:100px;overflow-y:auto;font-family:Consolas,monospace;font-size:11px;line-height:1.7;background:#060d14;border-radius:6px;padding:8px;border:1px solid #1a3a5a;}
.log-info{color:#aaa}.log-success{color:#4caf50}.log-warn{color:#ffc107}.log-error{color:#f44336}
.sig-BUY{color:#4caf50;font-weight:700}.sig-SELL{color:#f44336;font-weight:700}.sig-HOLD{color:#555}
.header-bar{background:#12151f;border-bottom:1px solid #2a2d3a;padding:10px 20px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
.search-box{position:relative;}
.search-results{position:absolute;top:100%;left:0;right:0;background:#1a1d27;border:1px solid #2a2d3a;border-radius:6px;z-index:999;max-height:180px;overflow-y:auto;display:none;}
.search-item{padding:7px 12px;cursor:pointer;font-size:12px;border-bottom:1px solid #2a2d3a;}
.search-item:hover{background:#212435;}
.pnl-pos{color:#4caf50;font-weight:600}.pnl-neg{color:#f44336;font-weight:600}
.live-set-box{background:#0a0d14;border-radius:6px;padding:10px 12px;border:1px solid #2a2d3a;margin-top:8px;}
::-webkit-scrollbar{width:4px;}::-webkit-scrollbar-track{background:#0f1117;}::-webkit-scrollbar-thumb{background:#2a2d3a;}
</style>
</head>
<body>
<div class="header-bar">
  <span style="font-size:15px;font-weight:700">NIFTY MULTI-BOT</span>
  <span id="hdr-count" style="color:#555;font-size:12px">0 symbols</span>
  <button class="btn-b" onclick="openLoginPopup()">Login Fyers</button>
  <button class="btn-g" id="btn-start-all" onclick="startAll()">Start All</button>
  <button class="btn-r" onclick="stopAll()">Stop All</button>
  <span id="login-status" style="font-size:12px;color:#4caf50;margin-left:4px"></span>
  <button id="theme-btn" onclick="toggleTheme()" style="margin-left:auto;background:transparent;border:1px solid #444;color:#aaa;border-radius:6px;padding:4px 12px;font-size:12px;cursor:pointer">Day</button>
</div>

<div class="container-fluid p-3">
<div class="row g-3">
<div class="col-12 col-xl-3">
  <div class="card">
    <div class="card-header">Add Symbol</div>
    <div class="p-3" style="display:flex;flex-direction:column;gap:9px">
      <div>
        <div style="font-size:11px;color:#666;margin-bottom:3px">Market</div>
        <select id="add-market">
          <option value="NSE_FO">NSE F&amp;O</option>
          <option value="NSE_CM">NSE Stocks</option>
          <option value="MCX_COM">MCX Commodity</option>
          <option value="BSE_CM">BSE Stocks</option>
        </select>
      </div>
      <div>
        <div style="font-size:11px;color:#666;margin-bottom:3px">Symbol Search</div>
        <div class="search-box">
          <input id="add-sym-input" placeholder="NIFTY / RELIANCE / GOLD..." autocomplete="off">
          <div id="add-sym-results" class="search-results"></div>
        </div>
        <div id="add-sym-selected" style="color:#2196f3;font-size:11px;margin-top:2px"></div>
      </div>
      <div class="row g-2">
        <div class="col-6"><div style="font-size:11px;color:#666;margin-bottom:3px">Strategy</div>
          <select id="add-strategy">
            <option value="ema_crossover">EMA Crossover</option>
            <option value="rsi_reversal">RSI Reversal</option>
            <option value="ema_rsi_combined">EMA+RSI Combined</option>
            <option value="bollinger_breakout">Bollinger Breakout</option>
            <option value="macd_crossover">MACD Crossover</option>
            <option value="ema_macd_combined">EMA+MACD Combined</option>
            <option value="vwap_ema9">VWAP+EMA9</option>
            <option value="vwap_ema9_rsi">VWAP+EMA9+RSI</option>
            <option value="five_condition_scalper">5-Condition Scalper</option>
            <option value="orb_915_930">ORB 9:15-9:30</option>
            <option value="gap_fade">Gap Fade</option>
          </select></div>
        <div class="col-6"><div style="font-size:11px;color:#666;margin-bottom:3px">Lot Size</div>
          <input type="number" id="add-lot" value="1" min="1"></div>
      </div>
      <div class="row g-2">
        <div class="col-6"><div style="font-size:11px;color:#666;margin-bottom:3px">Stop Loss %</div>
          <input type="number" id="add-sl" value="1.5" min="0.1" step="0.1"></div>
        <div class="col-6"><div style="font-size:11px;color:#666;margin-bottom:3px">Target %</div>
          <input type="number" id="add-tgt" value="2.5" min="0.1" step="0.1"></div>
      </div>
      <div class="row g-2">
        <div class="col-6"><div style="font-size:11px;color:#666;margin-bottom:3px">EMA Fast</div>
          <input type="number" id="add-ema-f" value="9" min="1"></div>
        <div class="col-6"><div style="font-size:11px;color:#666;margin-bottom:3px">EMA Slow</div>
          <input type="number" id="add-ema-s" value="21" min="1"></div>
      </div>
      <div class="row g-2">
        <div class="col-6"><div style="font-size:11px;color:#666;margin-bottom:3px">Max Lots</div>
          <input type="number" id="add-maxlots" value="1" min="1"></div>
        <div class="col-6"><div style="font-size:11px;color:#666;margin-bottom:3px">Mode</div>
          <select id="add-paper">
            <option value="true">Paper</option>
            <option value="false">Live</option>
          </select></div>
      </div>
      <div class="row g-2">
        <div class="col-6"><div style="font-size:11px;color:#666;margin-bottom:3px">Entry Time</div>
          <input type="time" id="add-entry" value="09:20"></div>
        <div class="col-6"><div style="font-size:11px;color:#666;margin-bottom:3px">Exit Time</div>
          <input type="time" id="add-exit" value="15:15"></div>
      </div>
      <button class="btn-g w-100" onclick="addSymbol()">+ Add Symbol</button>
      <div id="add-msg" style="font-size:11px;text-align:center;display:none"></div>
    </div>
  </div>
  <div class="card mt-3">
    <div class="card-header">Trade History</div>
    <div style="overflow-x:auto">
      <table style="width:100%;font-size:11px">
        <thead><tr style="color:#555">
          <th style="padding:5px 8px">Time</th>
          <th style="padding:5px 8px">Side</th>
          <th style="padding:5px 8px">Entry</th>
          <th style="padding:5px 8px">P&L</th>
        </tr></thead>
        <tbody id="trade-body"><tr><td colspan="4" style="text-align:center;color:#444;padding:12px">ਕੋਈ trade ਨਹੀਂ</td></tr></tbody>
      </table>
    </div>
  </div>
</div>

<div class="col-12 col-xl-9">

  <!-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ -->
  <!-- BANKBEES ML SCALPER PANEL              -->
  <!-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ -->
  <div class="bb-card">
    <div class="bb-header">
      <span style="font-size:14px;font-weight:800;color:#4fc3f7">🤖 BANKBEES ML SCALPER</span>
      <span id="bb-status-badge" class="badge-stop">STOPPED</span>
      <span id="bb-model-badge" class="badge-model-no">NO MODEL</span>
      <span style="color:#2a4a6a;font-size:11px">NSE:BANKBEES-EQ · CNC · 5min</span>

      <!-- ✅ QTY INPUT BOX — BB Header ਵਿੱਚ -->
      <div class="bb-qty-wrap" title="Order qty — 0 ਛੱਡੋ auto calculate ਲਈ">
        <label>QTY</label>
        <input type="number" id="bb-qty-input" class="bb-qty-input" value="10" min="1" max="9999"
               onkeydown="if(event.key==='Enter') bbSetQty()">
        <button class="bb-qty-set-btn" onclick="bbSetQty()">Set ✓</button>
      </div>

      <div style="margin-left:auto;display:flex;gap:8px;align-items:center">
        <span id="bb-last-check" style="color:#2a4a6a;font-size:11px">Last: —</span>
        <button id="bb-start-btn" class="btn-g" onclick="bbStart()">▶ Start</button>
        <button class="btn-r" onclick="bbStop()">■ Stop</button>
        <button class="btn-y" onclick="bbCloseTrade()" title="Manually mark trade closed">Close Trade</button>
      </div>
    </div>
    <div class="bb-body">
      <div class="row g-2 mb-3">
        <div class="col">
          <div class="bb-metric">
            <div class="bb-metric-lbl">Signal</div>
            <div id="bb-signal" class="bb-signal-hold">—</div>
          </div>
        </div>
        <div class="col">
          <div class="bb-metric">
            <div class="bb-metric-lbl">Probability</div>
            <div id="bb-prob" class="bb-metric-val" style="color:#00e676">—</div>
            <div class="bb-prob-bar"><div id="bb-prob-bar" class="bb-prob-fill" style="width:0%"></div></div>
          </div>
        </div>
        <div class="col">
          <div class="bb-metric">
            <div class="bb-metric-lbl">Entry</div>
            <div id="bb-entry" class="bb-metric-val" style="color:#e0e0e0">—</div>
          </div>
        </div>
        <div class="col">
          <div class="bb-metric">
            <div class="bb-metric-lbl">Target</div>
            <div id="bb-target" class="bb-metric-val" style="color:#00e676">—</div>
          </div>
        </div>
        <div class="col">
          <div class="bb-metric">
            <div class="bb-metric-lbl">Stop Loss</div>
            <div id="bb-sl" class="bb-metric-val" style="color:#f44336">—</div>
          </div>
        </div>
        <div class="col">
          <div class="bb-metric">
            <div class="bb-metric-lbl">Qty Used</div>
            <div id="bb-qty" class="bb-metric-val" style="color:#ffc107">—</div>
          </div>
        </div>
        <div class="col">
          <div class="bb-metric">
            <div class="bb-metric-lbl">Exit By</div>
            <div id="bb-exit-time" class="bb-metric-val" style="color:#90caf9;font-size:13px">—</div>
          </div>
        </div>
        <div class="col">
          <div class="bb-metric">
            <div class="bb-metric-lbl">In Trade</div>
            <div id="bb-in-trade" class="bb-metric-val" style="color:#ff9800">—</div>
          </div>
        </div>
      </div>
      <div class="bb-log-box" id="bb-log-box">
        <span style="color:#2a4a6a">BankBees Scalper ਚਾਲੂ ਕਰੋ...</span>
      </div>
    </div>
  </div>

  <!-- Existing symbols area -->
  <div id="symbols-area">
    <div id="empty-msg" style="color:#444;text-align:center;padding:60px;font-size:14px">
      &larr; ਖੱਬੇ ਪਾਸੇ symbol add ਕਰੋ
    </div>
  </div>
</div>

</div>
</div>

<div id="login-overlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:9999;align-items:center;justify-content:center">
  <div style="background:#1a1d27;border:1px solid #2a2d3a;border-radius:12px;padding:26px;width:500px;max-width:95vw">
    <div style="font-size:15px;font-weight:600;margin-bottom:14px">Fyers Login</div>
    <div style="font-size:12px;color:#888;margin-bottom:6px">Step 1 — ਇਸ URL ਤੇ click ਕਰੋ</div>
    <div id="login-url-box" onclick="openFyersUrl()" style="background:#12151f;border:1px solid #2a2d3a;border-radius:6px;padding:10px;font-size:11px;word-break:break-all;color:#2196f3;cursor:pointer;margin-bottom:10px">ਲੋਡ ਹੋ ਰਿਹਾ...</div>
    <button onclick="openFyersUrl()" class="btn-b" style="margin-bottom:12px">URL ਖੋਲ੍ਹੋ</button>
    <div style="font-size:12px;color:#888;margin-bottom:6px">Step 2 — Browser URL paste ਕਰੋ</div>
    <textarea id="login-token-input" placeholder="https://...?auth_code=eyJ..." style="background:#12151f;border:1px solid #2a2d3a;border-radius:6px;padding:10px;font-size:11px;color:#e0e0e0;width:100%;height:65px;resize:none;box-sizing:border-box;margin-bottom:10px"></textarea>
    <div style="display:flex;gap:10px">
      <button onclick="submitToken()" class="btn-g" style="flex:1">Submit</button>
      <button onclick="closeLoginPopup()" class="btn-r">Cancel</button>
    </div>
    <div id="login-msg" style="font-size:12px;margin-top:8px;display:none"></div>
  </div>
</div>

<script>
var addSymTimer = null;
var addSelectedSym = "";
var _fyersUrl = "";

function toggleTheme(){
  document.body.classList.toggle("light");
  document.getElementById("theme-btn").textContent = document.body.classList.contains("light") ? "Night" : "Day";
}

function nf(v, dec){
  dec = dec || 2;
  if(v === null || v === undefined || v === 0) return "—";
  return Number(v).toLocaleString("en-IN", {minimumFractionDigits:dec, maximumFractionDigits:dec});
}

// ── ✅ BankBees QTY Set ──────────────────────────────────────────────
function bbSetQty(){
  var inp = document.getElementById("bb-qty-input");
  var qty = parseInt(inp.value) || 10;
  qty = Math.max(qty, 1);
  inp.value = qty;
  fetch("/api/bankbees/set_qty", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({qty: qty})
  })
  .then(function(r){ return r.json(); })
  .then(function(d){
    if(d.ok){
      // ✅ feedback — input border flash ਹਰਾ
      inp.style.color = "#4caf50";
      setTimeout(function(){ inp.style.color = "#ffc107"; }, 1500);
    } else {
      alert("QTY set ਫੇਲ: " + d.msg);
    }
  });
}

// ── BankBees Controls ───────────────────────────────────────────────
function bbStart(){
  fetch("/api/bankbees/start", {method:"POST"})
    .then(function(r){ return r.json(); })
    .then(function(d){ if(!d.ok) alert("Error: " + d.msg); });
}
function bbStop(){
  fetch("/api/bankbees/stop", {method:"POST"});
}
function bbCloseTrade(){
  if(confirm("Manually mark BankBees trade as CLOSED?"))
    fetch("/api/bankbees/close_trade", {method:"POST"});
}

function renderBankBees(bb){
  if(!bb) return;

  // Status badge
  var sbadge = document.getElementById("bb-status-badge");
  sbadge.textContent  = bb.running ? "RUNNING" : "STOPPED";
  sbadge.className    = bb.running ? "badge-run" : "badge-stop";

  // Model badge
  var mbadge = document.getElementById("bb-model-badge");
  mbadge.textContent  = bb.model_ok ? "MODEL OK ✓" : "NO MODEL";
  mbadge.className    = bb.model_ok ? "badge-model-ok" : "badge-model-no";

  // Last check
  document.getElementById("bb-last-check").textContent = "Last: " + (bb.last_check || "—");

  // Signal
  var sigEl = document.getElementById("bb-signal");
  sigEl.textContent  = bb.signal || "—";
  sigEl.className    = bb.signal && bb.signal.indexOf("BUY") !== -1 ? "bb-signal-buy" : "bb-signal-hold";

  // Probability
  var prob = bb.prob || 0;
  document.getElementById("bb-prob").textContent = prob > 0 ? prob.toFixed(1) + "%" : "—";
  document.getElementById("bb-prob-bar").style.width = Math.min(prob, 100) + "%";

  // Trade details
  document.getElementById("bb-entry").textContent     = bb.entry  > 0 ? "₹" + nf(bb.entry)  : "—";
  document.getElementById("bb-target").textContent    = bb.target > 0 ? "₹" + nf(bb.target) : "—";
  document.getElementById("bb-sl").textContent        = bb.sl     > 0 ? "₹" + nf(bb.sl)     : "—";
  document.getElementById("bb-qty").textContent       = bb.qty    > 0 ? bb.qty               : "—";
  document.getElementById("bb-exit-time").textContent = bb.exit_time || "—";

  // In trade
  var itEl = document.getElementById("bb-in-trade");
  itEl.textContent  = bb.in_trade ? "YES 🔴" : "NO";
  itEl.style.color  = bb.in_trade ? "#f44336" : "#4caf50";

  // ✅ Qty input sync — ਜੇ server ਤੋਂ user_qty ਆਵੇ ਤਾਂ input update ਕਰੋ
  var qtyInp = document.getElementById("bb-qty-input");
  if(bb.user_qty && document.activeElement !== qtyInp){
    qtyInp.value = bb.user_qty;
  }

  // Logs
  var logBox = document.getElementById("bb-log-box");
  if(bb.logs && bb.logs.length){
    var html = "";
    bb.logs.forEach(function(l){
      html += "<div><span style='color:#2a4a6a'>" + l.t + "</span> <span class='log-" + l.lvl + "'>" + l.msg + "</span></div>";
    });
    logBox.innerHTML = html;
  }

  // Start btn
  document.getElementById("bb-start-btn").disabled = bb.running;
}

// ── Symbol Controls ─────────────────────────────────────────────────
function symId(sym){
  return "card_" + sym.replace(/[^a-zA-Z0-9]/g, "_");
}

document.getElementById("add-sym-input").addEventListener("input", function(){
  clearTimeout(addSymTimer);
  addSelectedSym = "";
  document.getElementById("add-sym-selected").textContent = "";
  var q = this.value.trim();
  if(q.length < 2){ document.getElementById("add-sym-results").style.display = "none"; return; }
  var mkt = document.getElementById("add-market").value;
  addSymTimer = setTimeout(function(){
    fetch("/api/search?q=" + encodeURIComponent(q) + "&market=" + mkt)
      .then(function(r){ return r.json(); })
      .then(function(list){
        var box = document.getElementById("add-sym-results");
        if(!list || !list.length){ box.style.display = "none"; return; }
        var html = "";
        list.forEach(function(s){
          html += "<div class='search-item' data-ticker='" + s.ticker + "' data-lot='" + s.lot + "' onclick='selectSym(this)'>";
          html += "<b>" + s.ticker + "</b> <small style='color:#888'>" + s.name + "</small>";
          html += "<span style='float:right;color:#666'>lot:" + s.lot + "</span></div>";
        });
        box.innerHTML = html;
        box.style.display = "block";
      }).catch(function(){});
  }, 400);
});

function selectSym(el){
  addSelectedSym = el.dataset.ticker;
  document.getElementById("add-sym-input").value = el.dataset.ticker;
  document.getElementById("add-lot").value = el.dataset.lot;
  document.getElementById("add-sym-results").style.display = "none";
  document.getElementById("add-sym-selected").textContent = "Selected: " + el.dataset.ticker;
  if(el.dataset.ticker.indexOf("MCX:") === 0){
    document.getElementById("add-entry").value = "10:00";
    document.getElementById("add-exit").value = "23:25";
  } else {
    document.getElementById("add-entry").value = "09:20";
    document.getElementById("add-exit").value = "15:15";
  }
}

document.addEventListener("click", function(e){
  if(!e.target.closest(".search-box"))
    document.getElementById("add-sym-results").style.display = "none";
});

function addSymbol(){
  var sym = addSelectedSym || document.getElementById("add-sym-input").value.trim();
  if(!sym){ alert("ਪਹਿਲਾਂ symbol ਚੁਣੋ"); return; }
  fetch("/api/add_symbol", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      symbol:        sym,
      strategy:      document.getElementById("add-strategy").value,
      lot_size:      parseInt(document.getElementById("add-lot").value) || 1,
      max_lots:      parseInt(document.getElementById("add-maxlots").value) || 1,
      stop_loss_pct: parseFloat(document.getElementById("add-sl").value) || 1.5,
      target_pct:    parseFloat(document.getElementById("add-tgt").value) || 2.5,
      ema_fast:      parseInt(document.getElementById("add-ema-f").value) || 9,
      ema_slow:      parseInt(document.getElementById("add-ema-s").value) || 21,
      entry_time:    document.getElementById("add-entry").value,
      exit_time:     document.getElementById("add-exit").value,
      paper_trade:   document.getElementById("add-paper").value === "true",
    })
  }).then(function(r){ return r.json(); }).then(function(d){
    var msg = document.getElementById("add-msg");
    msg.textContent = d.ok ? "Added: " + sym : "Error: " + d.msg;
    msg.style.color = d.ok ? "#4caf50" : "#f44336";
    msg.style.display = "block";
    setTimeout(function(){ msg.style.display = "none"; }, 3000);
    if(d.ok){ addSelectedSym = ""; document.getElementById("add-sym-input").value = ""; document.getElementById("add-sym-selected").textContent = ""; }
  });
}

function saveSetting(el){
  var sym   = el.dataset.sym;
  var field = el.dataset.field;
  var dtype = el.dataset.type || "string";
  var val   = dtype === "float" ? parseFloat(el.value) : dtype === "int" ? parseInt(el.value) : el.value;
  var payload = {symbol: sym};
  payload[field] = val;
  fetch("/api/update_symbol", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)})
    .then(function(r){ return r.json(); })
    .then(function(d){
      if(d.ok){ el.style.outline = "2px solid #4caf50"; setTimeout(function(){ el.style.outline = ""; }, 1000); }
      else { alert("Update failed: " + d.msg); }
    });
}

function startSymbol(sym){
  fetch("/api/start_symbol", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({symbol:sym})})
    .then(function(r){ return r.json(); })
    .then(function(d){ if(!d.ok) alert(d.msg); });
}
function stopSymbol(sym){
  fetch("/api/stop_symbol", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({symbol:sym})});
}
function removeSymbol(sym){
  if(!confirm("Remove " + sym + "?")) return;
  fetch("/api/remove_symbol", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({symbol:sym})});
}
function togglePaper(sym){
  fetch("/api/status").then(function(r){ return r.json(); }).then(function(d){
    var current = d.symbols[sym] ? d.symbols[sym].paper_trade : true;
    fetch("/api/update_symbol", {method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({symbol:sym, paper_trade:!current})});
  });
}

function startAll(){
  var btn = document.getElementById("btn-start-all");
  if(btn.disabled) return;
  btn.disabled = true; btn.textContent = "Checking..."; btn.style.opacity = "0.7";
  fetch("/api/status").then(function(r){ return r.json(); }).then(function(d){
    var toStart = Object.keys(d.symbols || {}).filter(function(s){ return !d.symbols[s].running; });
    if(toStart.length === 0){
      btn.textContent = "All Running";
      setTimeout(function(){ btn.textContent="Start All"; btn.disabled=false; btn.style.opacity="1"; }, 3000);
      return;
    }
    btn.textContent = "Starting...";
    fetch("/api/start_all", {method:"POST"})
      .then(function(r){ return r.json(); })
      .then(function(res){
        btn.textContent = (res.started||0) + " Started";
        setTimeout(function(){ btn.textContent="Start All"; btn.disabled=false; btn.style.opacity="1"; }, 3000);
      }).catch(function(){ btn.textContent="Start All"; btn.disabled=false; btn.style.opacity="1"; });
  });
}

function stopAll(){
  fetch("/api/status").then(function(r){ return r.json(); }).then(function(d){
    var syms    = d.symbols || {};
    var running = Object.keys(syms).filter(function(s){ return syms[s].running; });
    var hasPos  = Object.keys(syms).filter(function(s){ return syms[s].position; });
    if(running.length === 0){ alert("ਕੋਈ bot ਨਹੀਂ ਚੱਲ ਰਿਹਾ"); return; }
    var msg = running.length + " Bots ਬੰਦ ਕਰਨੇ ਹਨ";
    if(hasPos.length > 0) msg += "\n\nWARNING: " + hasPos.length + " open positions ਹਨ!";
    msg += "\n\nConfirm?";
    if(confirm(msg)) fetch("/api/stop_all", {method:"POST"});
  });
}

function openLoginPopup(){
  document.getElementById("login-overlay").style.display = "flex";
  document.getElementById("login-url-box").textContent = "ਲੋਡ ਹੋ ਰਿਹਾ...";
  document.getElementById("login-token-input").value = "";
  document.getElementById("login-msg").style.display = "none";
  fetch("/api/get_login_url").then(function(r){ return r.json(); }).then(function(d){
    if(d.ok){ _fyersUrl = d.url; document.getElementById("login-url-box").textContent = d.url; }
    else document.getElementById("login-url-box").textContent = "Error: " + d.msg;
  });
}
function openFyersUrl(){ if(_fyersUrl) window.open(_fyersUrl, "_blank"); }
function closeLoginPopup(){ document.getElementById("login-overlay").style.display = "none"; }

function submitToken(){
  var token = document.getElementById("login-token-input").value.trim();
  if(!token){ alert("Token paste ਕਰੋ"); return; }
  var msg = document.getElementById("login-msg");
  msg.textContent = "Login ਹੋ ਰਿਹਾ..."; msg.style.color = "#ffc107"; msg.style.display = "block";
  fetch("/api/submit_token", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({token:token})})
    .then(function(r){ return r.json(); }).then(function(d){
      if(d.ok){
        msg.textContent = "Login ਸਫ਼ਲ! " + d.name; msg.style.color = "#4caf50";
        document.getElementById("login-status").textContent = "✓ " + d.name;
        setTimeout(closeLoginPopup, 2000);
      } else { msg.textContent = "Error: " + d.msg; msg.style.color = "#f44336"; }
    });
}

function renderCard(sym, s, area){
  var id  = symId(sym);
  var run = s.running;
  var sig = s.signal || "HOLD";
  var el  = document.getElementById(id);
  if(!el){
    el = document.createElement("div");
    el.id = id;
    el.className = "sym-card";
    area.appendChild(el);
  }
  var posHtml = "<span style='color:#444;font-size:12px'>No position</span>";
  if(s.position){
    var qty  = s.position.filled_qty || 1;
    var pnl  = s.position.side === "BUY" ? (s.price - s.position.entry)*qty : (s.position.entry - s.price)*qty;
    var pc   = pnl >= 0 ? "pnl-pos" : "pnl-neg";
    var sl   = s.position.sl_price  || 0;
    var tgt  = s.position.tgt_price || 0;
    posHtml  = "<span style='color:" + (s.position.side==="BUY"?"#4caf50":"#f44336") + ";font-weight:700'>" + s.position.side + "</span>";
    posHtml += " <span style='color:#888;font-size:11px'>qty:" + qty + "</span>";
    posHtml += " @ <b>" + nf(s.position.entry) + "</b>";
    posHtml += " <span class='" + pc + "'>" + (pnl>=0?"+":"") + nf(pnl) + "</span>";
    if(sl  > 0) posHtml += "<br><span style='color:#f44336;font-size:11px'>SL:" + nf(sl) + "</span>";
    if(tgt > 0) posHtml += " <span style='color:#4caf50;font-size:11px'>TGT:" + nf(tgt) + "</span>";
  }
  var logsHtml = "";
  (s.logs||[]).forEach(function(l){
    logsHtml += "<div><span style='color:#444'>" + l.t + "</span> <span class='log-" + l.lvl + "'>" + l.msg + "</span></div>";
  });
  var strats = ["ema_crossover","rsi_reversal","ema_rsi_combined","bollinger_breakout","macd_crossover","ema_macd_combined","vwap_ema9","vwap_ema9_rsi","five_condition_scalper","orb_915_930","gap_fade"];
  var stratOpts = strats.map(function(st){
    return "<option value='" + st + "'" + (s.strategy===st?" selected":"") + ">" + st + "</option>";
  }).join("");
  el.innerHTML =
    "<div class='sym-header'>"
    + "<span style='font-weight:700;font-size:13px'>" + sym + "</span>"
    + "<span class='" + (run?"badge-run":"badge-stop") + "'>" + (run?"RUNNING":"STOPPED") + "</span>"
    + "<span class='" + (s.paper_trade?"badge-paper":"badge-live") + "'>" + (s.paper_trade?"PAPER":"LIVE") + "</span>"
    + "<span style='color:#555;font-size:11px'>" + (s.last_tick||"") + "</span>"
    + "<div style='margin-left:auto;display:flex;gap:6px'>"
    + "<button class='" + (s.paper_trade?"btn-y":"btn-b") + "' data-sym='" + sym + "' onclick='togglePaper(this.dataset.sym)'>" + (s.paper_trade?"PAPER":"LIVE") + "</button>"
    + (run
        ? "<button class='btn-r' data-sym='" + sym + "' onclick='stopSymbol(this.dataset.sym)'>Stop</button>"
        : "<button class='btn-g' data-sym='" + sym + "' onclick='startSymbol(this.dataset.sym)'>Start</button>")
    + "<button class='btn-r' style='padding:4px 8px' data-sym='" + sym + "' onclick='removeSymbol(this.dataset.sym)'>X</button>"
    + "</div></div>"
    + "<div class='sym-body'>"
    + "<div class='row g-2 mb-2'>"
    + "<div class='col'><div class='metric'><div class='metric-lbl'>Price</div><div class='metric-val'>" + (s.price>0?"&#8377;"+nf(s.price):"&#8212;") + "</div></div></div>"
    + "<div class='col'><div class='metric'><div class='metric-lbl'>Signal</div><div class='metric-val sig-" + sig + "'>" + sig + "</div></div></div>"
    + "<div class='col'><div class='metric'><div class='metric-lbl'>EMA Fast</div><div class='metric-val' style='color:#2196f3'>" + nf(s.ema_fast) + "</div></div></div>"
    + "<div class='col'><div class='metric'><div class='metric-lbl'>EMA Slow</div><div class='metric-val' style='color:#9c27b0'>" + nf(s.ema_slow) + "</div></div></div>"
    + "<div class='col'><div class='metric'><div class='metric-lbl'>RSI</div><div class='metric-val' style='color:#ff9800'>" + (s.rsi>0?s.rsi.toFixed(1):"&#8212;") + "</div></div></div>"
    + "<div class='col'><div class='metric'><div class='metric-lbl'>Scans</div><div class='metric-val' style='color:#607d8b'>" + (s.scan_no||0) + "</div></div></div>"
    + "</div>"
    + "<div class='row g-2 mb-2'>"
    + "<div class='col-12 col-md-4'><div style='background:#12151f;border-radius:6px;padding:8px 12px;font-size:12px;min-height:52px'><span style='color:#666'>Position: </span>" + posHtml + "</div></div>"
    + "<div class='col-12 col-md-8'><div class='log-box'>" + (logsHtml||"<span style='color:#333'>Bot ਚਾਲੂ ਕਰੋ...</span>") + "</div></div>"
    + "</div>"
    + "<div class='live-set-box'>"
    + "<div style='font-size:10px;color:#555;margin-bottom:8px;text-transform:uppercase'>Live Settings</div>"
    + "<div class='row g-2'>"
    + "<div class='col-12 col-md-3'><div style='font-size:10px;color:#666;margin-bottom:2px'>Strategy</div>"
    + "<select data-sym='" + sym + "' data-field='strategy' data-type='string' onchange='saveSetting(this)' style='font-size:11px;padding:3px'>" + stratOpts + "</select></div>"
    + "<div class='col'><div style='font-size:10px;color:#666;margin-bottom:2px'>SL%</div>"
    + "<input type='number' value='" + s.sl_pct + "' min='0.1' step='0.1' data-sym='" + sym + "' data-field='sl_pct' data-type='float' onchange='saveSetting(this)' style='font-size:11px;padding:3px'></div>"
    + "<div class='col'><div style='font-size:10px;color:#666;margin-bottom:2px'>TGT%</div>"
    + "<input type='number' value='" + s.tgt_pct + "' min='0.1' step='0.1' data-sym='" + sym + "' data-field='tgt_pct' data-type='float' onchange='saveSetting(this)' style='font-size:11px;padding:3px'></div>"
    + "<div class='col'><div style='font-size:10px;color:#666;margin-bottom:2px'>Lot</div>"
    + "<input type='number' value='" + s.lot_size + "' min='1' data-sym='" + sym + "' data-field='lot_size' data-type='int' onchange='saveSetting(this)' style='font-size:11px;padding:3px'></div>"
    + "<div class='col'><div style='font-size:10px;color:#666;margin-bottom:2px'>MaxLots</div>"
    + "<input type='number' value='" + s.max_lots + "' min='1' data-sym='" + sym + "' data-field='max_lots' data-type='int' onchange='saveSetting(this)' style='font-size:11px;padding:3px'></div>"
    + "<div class='col'><div style='font-size:10px;color:#666;margin-bottom:2px'>EMA F</div>"
    + "<input type='number' value='" + s.ema_fast_p + "' min='1' data-sym='" + sym + "' data-field='ema_fast' data-type='int' onchange='saveSetting(this)' style='font-size:11px;padding:3px'></div>"
    + "<div class='col'><div style='font-size:10px;color:#666;margin-bottom:2px'>EMA S</div>"
    + "<input type='number' value='" + s.ema_slow_p + "' min='1' data-sym='" + sym + "' data-field='ema_slow' data-type='int' onchange='saveSetting(this)' style='font-size:11px;padding:3px'></div>"
    + "</div></div>"
    + "</div>";
}

function poll(){
  fetch("/api/status")
    .then(function(r){ return r.json(); })
    .then(function(d){
      renderBankBees(d.bankbees);

      var syms = d.symbols || {};
      var keys = Object.keys(syms);
      document.getElementById("hdr-count").textContent = keys.length + " symbol" + (keys.length===1?"":"s");
      var area = document.getElementById("symbols-area");
      var emptyMsg = document.getElementById("empty-msg");
      if(keys.length === 0){
        if(!emptyMsg){
          area.innerHTML = "<div id='empty-msg' style='color:#444;text-align:center;padding:60px;font-size:14px'>&larr; ਖੱਬੇ ਪਾਸੇ symbol add ਕਰੋ</div>";
        }
        return;
      }
      if(emptyMsg) emptyMsg.remove();
      keys.forEach(function(sym){ renderCard(sym, syms[sym], area); });
      var cards = area.querySelectorAll(".sym-card");
      cards.forEach(function(el){
        var found = keys.some(function(sym){ return symId(sym) === el.id; });
        if(!found) el.remove();
      });
      var tb = document.getElementById("trade-body");
      if(d.trades && d.trades.length){
        var rows = "";
        d.trades.forEach(function(t){
          var pnl = parseFloat(t.pnl||0);
          rows += "<tr><td style='padding:4px 8px;color:#666'>" + (t.time||"") + "</td>"
            + "<td style='padding:4px 8px;color:" + (t.side==="BUY"?"#4caf50":"#f44336") + ";font-weight:600'>" + (t.side||"") + "</td>"
            + "<td style='padding:4px 8px'>&#8377;" + nf(t.entry) + "</td>"
            + "<td style='padding:4px 8px' class='" + (pnl>=0?"pnl-pos":"pnl-neg") + "'>" + (pnl>=0?"+":"") + nf(pnl) + "</td></tr>";
        });
        tb.innerHTML = rows;
      }
    }).catch(function(){});
}

poll();
setInterval(poll, 3000);
</script>
</body></html>"""

if __name__ == "__main__":
    try:
        cfg = load_config()
        sym = cfg["bot"]["index_symbol"]
        if sym not in symbols_state:
            symbols_state[sym] = default_sym_state(sym, cfg["bot"])
        print("✅ Config load ਹੋ ਗਈ")
        print("  Default symbol: " + sym)
    except Exception as e:
        print("Config error: " + str(e))

    load_symbols()

    try:
        import datetime as _dt
        from fyers_apiv3 import fyersModel as _fm
        cfg2 = load_config()
        tf   = cfg2["fyers"]["token_file"]
        if os.path.exists(tf):
            td = json.load(open(tf))
            today = _dt.datetime.now().strftime("%Y-%m-%d")
            if td.get("date") == today and td.get("access_token"):
                fyers = _fm.FyersModel(client_id=cfg2["fyers"]["client_id"],
                                       token=td["access_token"], log_path="")
                profile = fyers.get_profile()
                name = profile.get("data",{}).get("name","") if profile.get("s")=="ok" else ""
                if name: print("  Auto-login: " + name)
    except Exception as e:
        print("  Auto-login skip: " + str(e))

    print("\n" + "="*45)
    print("  Nifty Multi-Bot GUI v3.2 — BankBees User QTY")
    print("="*45)
    print("  http://localhost:5000")
    print("="*45 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
