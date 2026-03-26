//+------------------------------------------------------------------+
//|  MT5_Bot_mk2_EA.mq5                                              |
//|  Bidirectional ZMQ bridge for MT5_Bot_mk2                        |
//|                                                                  |
//|  ZMQ topology (EA binds, Python connects):                       |
//|    EA  PUSH binds   tcp://*:5557  → Python PULL connects 5557    |
//|    EA  PULL connects tcp://127.0.0.1:5558 ← Python PUSH binds   |
//|                                                                  |
//|  Requires: mql-zmq library (github.com/dingmaotu/mql-zmq)       |
//|    Include\Zmq\Zmq.mqh + libzmq.dll in Libraries\               |
//+------------------------------------------------------------------+
#property strict

#include <Trade/Trade.mqh>
#include <Trade/PositionInfo.mqh>
#include <Zmq/Zmq.mqh>

// ── Parameters ───────────────────────────────────────────────────────────────
input int    PushPort      = 5557;   // EA → Python (EA binds this)
input string PythonHost    = "127.0.0.1";
input int    PullPort      = 5558;   // Python → EA (Python binds, EA connects)
input int    TimerMs       = 100;
input int    MaxHistoryBars = 300000;
input bool   DebugMode     = false;

// ── Globals ──────────────────────────────────────────────────────────────────
Context   g_ctx;
Socket*   g_push = NULL;   // EA → Python  (PUSH, binds)
Socket*   g_pull = NULL;   // Python → EA  (PULL, connects)

CTrade        g_trade;
CPositionInfo g_pos;

ulong    g_magic   = 20250002;
datetime g_lastHB  = 0;
ulong    g_lastHistTkt = 0;

//+------------------------------------------------------------------+
//| OnInit                                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   g_ctx.setBlocky(false);

   // EA PUSH socket — binds so Python can connect to it
   g_push = new Socket(g_ctx, ZMQ_PUSH);
   g_push.setSendHighWaterMark(1000);
   g_push.setLinger(0);
   string pushAddr = "tcp://*:" + IntegerToString(PushPort);
   if(!g_push.bind(pushAddr))
   {
      Print("ERROR: Cannot bind PUSH socket on ", pushAddr,
            "  err=", Zmq::errorMessage());
      return INIT_FAILED;
   }

   // EA PULL socket — connects to Python's PUSH bind address
   g_pull = new Socket(g_ctx, ZMQ_PULL);
   g_pull.setReceiveHighWaterMark(1000);
   g_pull.setLinger(0);
   string pullAddr = "tcp://" + PythonHost + ":" + IntegerToString(PullPort);
   if(!g_pull.connect(pullAddr))
   {
      Print("ERROR: Cannot connect PULL socket to ", pullAddr,
            "  err=", Zmq::errorMessage());
      return INIT_FAILED;
   }

   g_trade.SetExpertMagicNumber(g_magic);
   g_trade.SetDeviationInPoints(10);
   g_trade.LogLevel(LOG_LEVEL_ERRORS);

   EventSetMillisecondTimer(TimerMs);
   Print("MT5_Bot_mk2_EA ready | PUSH binds ", pushAddr,
         " | PULL connects ", pullAddr);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| OnDeinit                                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   if(g_push != NULL) { delete g_push; g_push = NULL; }
   if(g_pull != NULL) { delete g_pull; g_pull = NULL; }
   Print("MT5_Bot_mk2_EA stopped. reason=", reason);
}

//+------------------------------------------------------------------+
//| OnTimer — main loop                                               |
//+------------------------------------------------------------------+
void OnTimer()
{
   // Process all pending inbound commands
   ProcessInbound();

   // Push market data snapshot
   PushAllTicks();

   // Periodic account info + closed-trade publish
   if(TimeCurrent() - g_lastHB >= 10)
   {
      PushAccountInfo();
      PublishClosedTrades();
      PushOpenPositions();   // lets Python reconcile open trades every 10s
      g_lastHB = TimeCurrent();
   }
}

//+------------------------------------------------------------------+
//| OnTick — fast tick push for current chart symbol                 |
//+------------------------------------------------------------------+
void OnTick()
{
   PushTick(Symbol());
}

//+------------------------------------------------------------------+
//| SEND helper                                                       |
//+------------------------------------------------------------------+
void SendMsg(string json)
{
   if(g_push == NULL) return;
   ZmqMsg zm(json);
   g_push.send(zm, true);
   if(DebugMode) Print("PUSH: ", json);
}

//+------------------------------------------------------------------+
//| RECV helper — returns false when queue is empty                  |
//+------------------------------------------------------------------+
bool RecvMsg(string &out)
{
   if(g_pull == NULL) return false;
   ZmqMsg msg;
   if(!g_pull.recv(msg, true)) return false;
   out = msg.getData();
   if(DebugMode) Print("PULL: ", out);
   return true;
}

//+------------------------------------------------------------------+
//| Process all inbound messages (non-blocking)                      |
//+------------------------------------------------------------------+
void ProcessInbound()
{
   string raw;
   int cap = 30;
   while(cap-- > 0 && RecvMsg(raw))
   {
      string mtype = JsonGet(raw, "type");
      if(mtype == "TRADE")     ExecTrade(raw);
      else if(mtype == "CMD")  ExecCmd(raw);
      else if(mtype == "HEARTBEAT") SendMsg("{\"type\":\"HEARTBEAT\"}");
   }
}

//+------------------------------------------------------------------+
//| Push tick for a symbol                                           |
//+------------------------------------------------------------------+
void PushTick(string symbol)
{
   MqlTick tick;
   if(!SymbolInfoTick(symbol, tick)) return;
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   double spread = (tick.ask - tick.bid) * MathPow(10.0, digits - 1);
   string msg = "{\"type\":\"TICK\","
              + "\"symbol\":\"" + symbol + "\","
              + "\"bid\":"    + DoubleToString(tick.bid, digits) + ","
              + "\"ask\":"    + DoubleToString(tick.ask, digits) + ","
              + "\"spread\":" + DoubleToString(spread, 1) + ","
              + "\"volume\":" + IntegerToString((long)tick.volume) + ","
              + "\"ts\":"     + IntegerToString((long)tick.time)
              + "}";
   SendMsg(msg);
}

//+------------------------------------------------------------------+
//| Push ticks for all market-watch symbols (capped at 50)           |
//+------------------------------------------------------------------+
void PushAllTicks()
{
   int total = SymbolsTotal(true);
   for(int i = 0; i < total && i < 50; i++)
   {
      string sym = SymbolName(i, true);
      if(StringLen(sym) > 0) PushTick(sym);
   }
}

//+------------------------------------------------------------------+
//| Push account info                                                |
//+------------------------------------------------------------------+
void PushAccountInfo()
{
   string msg = "{\"type\":\"ACCOUNT_INFO\","
              + "\"balance\":"     + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2) + ","
              + "\"equity\":"      + DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY),  2) + ","
              + "\"margin\":"      + DoubleToString(AccountInfoDouble(ACCOUNT_MARGIN),  2) + ","
              + "\"free_margin\":" + DoubleToString(AccountInfoDouble(ACCOUNT_MARGIN_FREE), 2) + ","
              + "\"profit\":"      + DoubleToString(AccountInfoDouble(ACCOUNT_PROFIT),  2) + ","
              + "\"currency\":\""  + AccountInfoString(ACCOUNT_CURRENCY) + "\","
              + "\"leverage\":"    + IntegerToString((long)AccountInfoInteger(ACCOUNT_LEVERAGE))
              + "}";
   SendMsg(msg);
}

//+------------------------------------------------------------------+
//| Publish recently closed trades (all mk2 magic range)             |
//| Magic range: 20250001 – 20269999 covers all stacked trade magics |
//+------------------------------------------------------------------+
void PublishClosedTrades()
{
   datetime from = TimeCurrent() - 86400;
   if(!HistorySelect(from, TimeCurrent())) return;
   int total = HistoryDealsTotal();
   for(int i = total - 1; i >= 0; i--)
   {
      ulong tkt = HistoryDealGetTicket(i);
      if(tkt == 0 || tkt <= g_lastHistTkt) break;
      long  dealMagic = HistoryDealGetInteger(tkt, DEAL_MAGIC);
      // Accept any mk2 magic (20250001–20269999) OR magic=0 (manual close)
      bool isMk2 = (dealMagic >= 20250001 && dealMagic < 20270000) || dealMagic == 0;
      if(!isMk2) continue;
      if(HistoryDealGetInteger(tkt, DEAL_ENTRY) != DEAL_ENTRY_OUT) continue;
      double profit = HistoryDealGetDouble(tkt, DEAL_PROFIT)
                    + HistoryDealGetDouble(tkt, DEAL_SWAP)
                    + HistoryDealGetDouble(tkt, DEAL_COMMISSION);
      string sym    = HistoryDealGetString(tkt, DEAL_SYMBOL);
      ulong  posId  = (ulong)HistoryDealGetInteger(tkt, DEAL_POSITION_ID);
      string msg = "{\"type\":\"TRADE_CLOSED\","
                 + "\"symbol\":\""  + sym + "\","
                 + "\"ticket\":"    + IntegerToString((long)posId) + ","
                 + "\"magic\":"     + IntegerToString(dealMagic) + ","
                 + "\"profit\":"    + DoubleToString(profit, 2)
                 + "}";
      SendMsg(msg);
      g_lastHistTkt = tkt;
      break; // one per cycle to avoid flooding
   }
}

//+------------------------------------------------------------------+
//| Push all currently open positions for Python reconciliation       |
//+------------------------------------------------------------------+
void PushOpenPositions()
{
   string arr = "[";
   bool first = true;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(!g_pos.SelectByIndex(i)) continue;
      long magic = (long)g_pos.Magic();
      if(magic != 0 && (magic < 20250001 || magic >= 20270000)) continue;
      if(!first) arr += ",";
      arr += "{"
           + "\"symbol\":\""    + g_pos.Symbol()                        + "\","
           + "\"ticket\":"      + IntegerToString((long)g_pos.Ticket())  + ","
           + "\"magic\":"       + IntegerToString(magic)                 + ","
           + "\"direction\":\"" + (g_pos.PositionType()==POSITION_TYPE_BUY ? "BUY" : "SELL") + "\","
           + "\"lot\":"         + DoubleToString(g_pos.Volume(), 2)      + ","
           + "\"open_price\":"  + DoubleToString(g_pos.PriceOpen(), 5)   + ","
           + "\"sl\":"          + DoubleToString(g_pos.StopLoss(),  5)   + ","
           + "\"tp\":"          + DoubleToString(g_pos.TakeProfit(), 5)  + ","
           + "\"profit\":"      + DoubleToString(g_pos.Profit(), 2)
           + "}";
      first = false;
   }
   arr += "]";
   SendMsg("{\"type\":\"POSITIONS_SYNC\",\"positions\":" + arr + "}");
}

//+------------------------------------------------------------------+
//| Execute a TRADE command from Python                              |
//+------------------------------------------------------------------+
void ExecTrade(string json)
{
   string symbol = JsonGet(json, "symbol");
   string action = JsonGet(json, "action");
   double lot    = StringToDouble(JsonGet(json, "lot"));
   double sl     = StringToDouble(JsonGet(json, "sl"));
   double tp     = StringToDouble(JsonGet(json, "tp"));
   string comment= JsonGet(json, "comment");
   ulong  magic  = (ulong)StringToInteger(JsonGet(json, "magic"));
   if(magic == 0) magic = g_magic;

   if(StringLen(symbol) == 0 || lot <= 0) { Print("Invalid trade: ", json); return; }

   g_trade.SetExpertMagicNumber(magic);
   MqlTick tick;
   SymbolInfoTick(symbol, tick);

   bool ok = false;
   if(action == "BUY")
      ok = g_trade.Buy(lot, symbol, tick.ask, sl, tp, comment);
   else if(action == "SELL")
      ok = g_trade.Sell(lot, symbol, tick.bid, sl, tp, comment);
   else if(action == "CLOSE")
      CloseBySymbol(symbol, (long)magic);
   else if(action == "MODIFY")
      ModifyOpenTrades(symbol, (long)magic, sl, tp);

   if(ok)
   {
      ulong ticket = g_trade.ResultOrder();
      SendMsg("{\"type\":\"TRADE_CONFIRM\","
            + "\"symbol\":\""  + symbol + "\","
            + "\"ticket\":"    + IntegerToString((long)ticket) + ","
            + "\"magic\":"     + IntegerToString((long)magic) + ","
            + "\"price\":"     + DoubleToString(g_trade.ResultPrice(), 5)
            + "}");
   }
   else if(action == "BUY" || action == "SELL")
      Print("Trade failed [", g_trade.ResultRetcode(), "]: ",
            g_trade.ResultRetcodeDescription(), " | ", json);
}

//+------------------------------------------------------------------+
//| Close all positions for symbol with given magic                  |
//+------------------------------------------------------------------+
void CloseBySymbol(string symbol, long magic)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!g_pos.SelectByIndex(i)) continue;
      if(g_pos.Symbol() != symbol) continue;
      if(magic > 0 && (long)g_pos.Magic() != magic) continue;
      double profit = g_pos.Profit();
      ulong tkt = g_pos.Ticket();
      if(g_trade.PositionClose(tkt))
      {
         SendMsg("{\"type\":\"TRADE_CLOSED\","
               + "\"symbol\":\""  + symbol + "\","
               + "\"ticket\":"    + IntegerToString((long)tkt) + ","
               + "\"profit\":"    + DoubleToString(profit, 2)
               + "}");
      }
   }
}

//+------------------------------------------------------------------+
//| Modify SL/TP on matching open positions                          |
//+------------------------------------------------------------------+
void ModifyOpenTrades(string symbol, long magic, double sl, double tp)
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(!g_pos.SelectByIndex(i)) continue;
      if(g_pos.Symbol() != symbol) continue;
      if(magic > 0 && (long)g_pos.Magic() != magic) continue;
      g_trade.PositionModify(g_pos.Ticket(), sl, tp);
   }
}

//+------------------------------------------------------------------+
//| Execute control commands                                         |
//+------------------------------------------------------------------+
void ExecCmd(string json)
{
   string cmd = JsonGet(json, "cmd");

   if(cmd == "HANDSHAKE")
   {
      Print("Handshake from Python mk2 v", JsonGet(json, "version"));
      SendMsg("{\"type\":\"HANDSHAKE_ACK\",\"status\":\"ok\"}");
      PushAccountInfo();
   }
   else if(cmd == "EXPORT_HISTORICAL")
   {
      string symbol = JsonGet(json, "symbol");
      string tf_str = JsonGet(json, "timeframe");
      int    years  = (int)StringToInteger(JsonGet(json, "years"));
      ExportHistoricalData(symbol, tf_str, years);
   }
   else if(cmd == "SYMBOL_INFO")
   {
      PushSymbolInfo(JsonGet(json, "symbol"));
   }
   else if(cmd == "GET_ACCOUNT_INFO")
   {
      PushAccountInfo();
   }
   else if(cmd == "SUBSCRIBE_TICKS")
   {
      SymbolSelect(JsonGet(json, "symbol"), true);
   }
   else if(cmd == "GET_MARKETWATCH")
   {
      PushMarketWatchList();
   }
   else if(cmd == "GET_POSITIONS")
   {
      PushOpenPositions();
   }
}

//+------------------------------------------------------------------+
//| Push all Market Watch symbols as a JSON array                    |
//| Python uses this to resolve canonical names to broker-exact names|
//+------------------------------------------------------------------+
void PushMarketWatchList()
{
   string arr = "[";
   bool first = true;
   int total = SymbolsTotal(true);   // true = Market Watch only
   for(int i = 0; i < total; i++)
   {
      string sym = SymbolName(i, true);
      if(StringLen(sym) == 0) continue;
      // Only include tradeable symbols
      if(SymbolInfoInteger(sym, SYMBOL_TRADE_MODE) == SYMBOL_TRADE_MODE_DISABLED)
         continue;
      if(!first) arr += ",";
      arr += "\"" + sym + "\"";
      first = false;
   }
   arr += "]";
   SendMsg("{\"type\":\"MARKETWATCH_LIST\",\"symbols\":" + arr + "}");
   Print("MarketWatch list sent — ", total, " symbols");
}

//+------------------------------------------------------------------+
//| Export historical OHLCV to CSV in MT5 common files folder        |
//+------------------------------------------------------------------+
void ExportHistoricalData(string symbol, string tf_str, int years)
{
   ENUM_TIMEFRAMES tf = StringToTF(tf_str);
   if(!SymbolSelect(symbol, true))
   {
      Print("SymbolSelect failed: ", symbol);
      SendExportDone(symbol, false);
      return;
   }

   datetime end_time   = TimeCurrent();
   datetime start_time = end_time - (datetime)(years * 365 * 86400);

   MqlRates rates[];
   int copied = CopyRates(symbol, tf, start_time, end_time, rates);
   if(copied <= 0)
   {
      Print("CopyRates failed for ", symbol, " err=", GetLastError());
      SendExportDone(symbol, false);
      return;
   }

   // Write to MT5 common files (accessible from Python via MQL5\Files\Common)
   string path = "MT5_Bot_mk2\\data\\raw\\" + symbol + ".csv";
   // FILE_ANSI forces UTF-8 output — without it MQL5 writes UTF-16 LE
   int fh = FileOpen(path, FILE_WRITE | FILE_CSV | FILE_COMMON | FILE_ANSI, ',');
   if(fh == INVALID_HANDLE)
   {
      Print("FileOpen failed: ", path, " err=", GetLastError());
      SendExportDone(symbol, false);
      return;
   }

   FileWrite(fh, "timestamp", "open", "high", "low", "close", "tick_volume", "spread");

   MqlTick lastTick;
   SymbolInfoTick(symbol, lastTick);
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   double spread_val = (lastTick.ask - lastTick.bid) * MathPow(10.0, digits - 1);

   for(int i = 0; i < copied; i++)
   {
      FileWrite(fh,
         TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES|TIME_SECONDS),
         DoubleToString(rates[i].open,  digits),
         DoubleToString(rates[i].high,  digits),
         DoubleToString(rates[i].low,   digits),
         DoubleToString(rates[i].close, digits),
         IntegerToString(rates[i].tick_volume),
         DoubleToString(spread_val, 1)
      );
   }
   FileClose(fh);
   Print("Exported ", copied, " bars for ", symbol, " → ", path);
   SendExportDone(symbol, true);
}

void SendExportDone(string symbol, bool success)
{
   SendMsg("{\"type\":\"EXPORT_DONE\","
         + "\"symbol\":\""  + symbol + "\","
         + "\"success\":"   + (success ? "true" : "false")
         + "}");
}

//+------------------------------------------------------------------+
//| Push symbol info                                                 |
//+------------------------------------------------------------------+
void PushSymbolInfo(string symbol)
{
   if(!SymbolSelect(symbol, true)) return;
   SendMsg("{\"type\":\"SYMBOL_INFO\","
         + "\"symbol\":\""   + symbol + "\","
         + "\"digits\":"     + IntegerToString((long)SymbolInfoInteger(symbol, SYMBOL_DIGITS)) + ","
         + "\"point\":"      + DoubleToString(SymbolInfoDouble(symbol, SYMBOL_POINT), 8) + ","
         + "\"min_lot\":"    + DoubleToString(SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN), 2) + ","
         + "\"max_lot\":"    + DoubleToString(SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX), 2) + ","
         + "\"lot_step\":"   + DoubleToString(SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP), 2)
         + "}");
}

//+------------------------------------------------------------------+
//| Timeframe string to ENUM                                         |
//+------------------------------------------------------------------+
ENUM_TIMEFRAMES StringToTF(string tf)
{
   if(tf == "M1")  return PERIOD_M1;
   if(tf == "M5")  return PERIOD_M5;
   if(tf == "M15") return PERIOD_M15;
   if(tf == "M30") return PERIOD_M30;
   if(tf == "H1")  return PERIOD_H1;
   if(tf == "H4")  return PERIOD_H4;
   if(tf == "D1")  return PERIOD_D1;
   return PERIOD_M1;
}

//+------------------------------------------------------------------+
//| Minimal JSON field extractor                                     |
//+------------------------------------------------------------------+
string JsonGet(string json, string key)
{
   string search = "\"" + key + "\":";
   int pos = StringFind(json, search);
   if(pos < 0) return "";
   int start = pos + StringLen(search);
   // skip whitespace
   while(start < StringLen(json) && StringGetCharacter(json, start) == ' ') start++;
   if(start >= StringLen(json)) return "";
   ushort first = StringGetCharacter(json, start);
   if(first == '"')
   {
      start++;
      int end = StringFind(json, "\"", start);
      return (end < 0) ? "" : StringSubstr(json, start, end - start);
   }
   else
   {
      string val = "";
      for(int i = start; i < StringLen(json); i++)
      {
         ushort c = StringGetCharacter(json, i);
         if(c == ',' || c == '}' || c == ']' || c == ' ' || c == '\n' || c == '\r') break;
         val += CharToString((uchar)c);
      }
      return val;
   }
}
//+------------------------------------------------------------------+
