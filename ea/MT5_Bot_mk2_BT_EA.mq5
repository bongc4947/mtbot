//+------------------------------------------------------------------+
//|  MT5_Bot_mk2_BT_EA.mq5                                           |
//|  Strategy Tester / Backtest EA — NO ZMQ dependency               |
//|                                                                  |
//|  Use this EA exclusively in MT5 Strategy Tester.                 |
//|  For live trading use MT5_Bot_mk2_EA.mq5 instead.               |
//|                                                                  |
//|  Signal:   Fast EMA crosses Slow EMA (configurable periods)      |
//|  Trailing: Pip-based, activates once position is in profit       |
//|  SL/TP:    Clamped against SYMBOL_TRADE_STOPS_LEVEL (same as     |
//|            live EA) — validates the stop-management logic        |
//|                                                                  |
//|  Backtest analysis v2 fixes (2026-03-28):                        |
//|  1. Defaults scaled for volatile assets (GOLD/indices/crypto)    |
//|     — original 25-pip SL = only 2.5 pts on GOLD, hit instantly   |
//|  2. Close opposite direction before opening new signal           |
//|     — prevents simultaneous hedge positions both SL-hit           |
//|  3. Trailing minimum-move filter (≥25% of trail distance)        |
//|     — eliminates per-tick modify spam (was 27 mods/trade avg)    |
//+------------------------------------------------------------------+
#property strict

#include <Trade/Trade.mqh>
#include <Trade/PositionInfo.mqh>

// ── Parameters ───────────────────────────────────────────────────────────────
// Pip note: 1 pip = 0.0001 for forex, 0.01 for JPY, 0.10 for GOLD/indices.
// Defaults below are tuned for GOLD on M5+. Scale down for tight-spread forex.
input int    BT_FastMA    = 5;     // Fast EMA period
input int    BT_SlowMA    = 20;    // Slow EMA period
input double BT_Lot       = 0.01;  // Fixed lot size
input double BT_SL_Pips   = 150.0; // Initial SL in pips  (150 × 0.10 = 15 pts GOLD)
input double BT_TP_Pips   = 350.0; // Initial TP in pips  (350 × 0.10 = 35 pts GOLD)
input double BT_Trail_Act = 80.0;  // Trail activates after N pips profit (80 × 0.10 = 8 pts)
input double BT_Trail_Dist= 60.0;  // SL distance once trailing (60 × 0.10 = 6 pts)

// ── Globals ──────────────────────────────────────────────────────────────────
CTrade        g_trade;
CPositionInfo g_pos;

double   g_pip          = 0.0;
int      g_h_fast       = INVALID_HANDLE;
int      g_h_slow       = INVALID_HANDLE;
ulong    g_magic        = 88880001;

//+------------------------------------------------------------------+
//| OnInit                                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   // Derive pip size from symbol digits (standard MT5 5-digit convention)
   int digits = (int)SymbolInfoInteger(Symbol(), SYMBOL_DIGITS);
   if(digits == 3 || digits == 5)   g_pip = MathPow(10.0, -(digits - 1));
   else if(digits == 2)             g_pip = MathPow(10.0, -(digits - 1));
   else                             g_pip = SymbolInfoDouble(Symbol(), SYMBOL_POINT) * 10;

   g_h_fast = iMA(Symbol(), PERIOD_CURRENT, BT_FastMA, 0, MODE_EMA, PRICE_CLOSE);
   g_h_slow = iMA(Symbol(), PERIOD_CURRENT, BT_SlowMA, 0, MODE_EMA, PRICE_CLOSE);
   if(g_h_fast == INVALID_HANDLE || g_h_slow == INVALID_HANDLE)
   {
      Print("ERROR: iMA handle creation failed");
      return INIT_FAILED;
   }

   g_trade.SetExpertMagicNumber(g_magic);
   g_trade.SetDeviationInPoints(30);
   g_trade.LogLevel(LOG_LEVEL_ERRORS);

   Print("MT5_Bot_mk2_BT_EA | symbol=", Symbol(),
         " pip=", g_pip,
         " FastMA=", BT_FastMA, " SlowMA=", BT_SlowMA,
         " SL=", BT_SL_Pips * g_pip, "pts",
         " TP=", BT_TP_Pips * g_pip, "pts",
         " Trail@", BT_Trail_Act * g_pip, "pts/", BT_Trail_Dist * g_pip, "pts");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| OnDeinit                                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_h_fast != INVALID_HANDLE) { IndicatorRelease(g_h_fast); g_h_fast = INVALID_HANDLE; }
   if(g_h_slow != INVALID_HANDLE) { IndicatorRelease(g_h_slow); g_h_slow = INVALID_HANDLE; }
}

//+------------------------------------------------------------------+
//| OnTick                                                            |
//+------------------------------------------------------------------+
void OnTick()
{
   CheckSignal();
   UpdateTrailing();
}

//+------------------------------------------------------------------+
//| Close all open positions in the given direction                  |
//+------------------------------------------------------------------+
void CloseDirection(ENUM_POSITION_TYPE dir)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(!g_pos.SelectByIndex(i)) continue;
      if(g_pos.Symbol() != Symbol()) continue;
      if(g_pos.PositionType() == dir)
         g_trade.PositionClose(g_pos.Ticket());
   }
}

//+------------------------------------------------------------------+
//| EMA crossover signal → close opposite, open new                 |
//| Only ONE position open at a time (direction flip on crossover).  |
//+------------------------------------------------------------------+
void CheckSignal()
{
   if(Bars(Symbol(), PERIOD_CURRENT) < BT_SlowMA + 3) return;

   // CopyBuffer: start_pos=1 skips current incomplete bar
   // buf[0]=shift1 (last closed bar), buf[1]=shift2
   double fast_buf[2], slow_buf[2];
   if(CopyBuffer(g_h_fast, 0, 1, 2, fast_buf) < 2) return;
   if(CopyBuffer(g_h_slow, 0, 1, 2, slow_buf) < 2) return;

   double fast1 = fast_buf[0];   // shift 1
   double fast2 = fast_buf[1];   // shift 2
   double slow1 = slow_buf[0];
   double slow2 = slow_buf[1];

   bool crossUp   = (fast2 <= slow2) && (fast1 > slow1);
   bool crossDown = (fast2 >= slow2) && (fast1 < slow1);
   if(!crossUp && !crossDown) return;

   // Count open positions per direction
   int buys = 0, sells = 0;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(!g_pos.SelectByIndex(i)) continue;
      if(g_pos.Symbol() != Symbol()) continue;
      if(g_pos.PositionType() == POSITION_TYPE_BUY)  buys++;
      else                                            sells++;
   }

   MqlTick tick;
   if(!SymbolInfoTick(Symbol(), tick)) return;

   int    digits   = (int)SymbolInfoInteger(Symbol(), SYMBOL_DIGITS);
   double point    = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
   int    stops_lv = (int)SymbolInfoInteger(Symbol(), SYMBOL_TRADE_STOPS_LEVEL);
   double min_dist = MathMax(stops_lv + 5, 20) * point;
   double sl_dist  = MathMax(BT_SL_Pips * g_pip, min_dist);
   double tp_dist  = MathMax(BT_TP_Pips * g_pip, min_dist * 2);

   if(crossUp && buys == 0)
   {
      // Close any open SELL before flipping to BUY
      if(sells > 0) CloseDirection(POSITION_TYPE_SELL);

      double sl = NormalizeDouble(tick.bid - sl_dist, digits);
      double tp = NormalizeDouble(tick.ask + tp_dist, digits);
      if(g_trade.Buy(BT_Lot, Symbol(), tick.ask, sl, tp, "bt_buy"))
         Print("[BT] BUY  @ ", tick.ask, "  SL=", sl, "  TP=", tp);
   }
   else if(crossDown && sells == 0)
   {
      // Close any open BUY before flipping to SELL
      if(buys > 0) CloseDirection(POSITION_TYPE_BUY);

      double sl = NormalizeDouble(tick.ask + sl_dist, digits);
      double tp = NormalizeDouble(tick.bid - tp_dist, digits);
      if(g_trade.Sell(BT_Lot, Symbol(), tick.bid, sl, tp, "bt_sell"))
         Print("[BT] SELL @ ", tick.bid, "  SL=", sl, "  TP=", tp);
   }
}

//+------------------------------------------------------------------+
//| Pip-based trailing stop                                          |
//| Activates once position profit >= BT_Trail_Act pips.            |
//| Only modifies when new SL improves by >= 25% of trail distance  |
//| — prevents per-tick modify spam.                                 |
//+------------------------------------------------------------------+
void UpdateTrailing()
{
   MqlTick tick;
   if(!SymbolInfoTick(Symbol(), tick)) return;

   int    digits   = (int)SymbolInfoInteger(Symbol(), SYMBOL_DIGITS);
   double point    = SymbolInfoDouble(Symbol(), SYMBOL_POINT);
   int    stops_lv = (int)SymbolInfoInteger(Symbol(), SYMBOL_TRADE_STOPS_LEVEL);
   double min_dist = MathMax(stops_lv + 5, 20) * point;
   double trail    = MathMax(BT_Trail_Dist * g_pip, min_dist);
   double act      = BT_Trail_Act * g_pip;
   double min_step = trail * 0.25;   // only modify if SL improves ≥25% of trail dist

   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(!g_pos.SelectByIndex(i)) continue;
      if(g_pos.Symbol() != Symbol()) continue;

      double cur_sl = g_pos.StopLoss();
      double cur_tp = g_pos.TakeProfit();
      double new_sl = cur_sl;

      if(g_pos.PositionType() == POSITION_TYPE_BUY)
      {
         if(tick.bid - g_pos.PriceOpen() < act) continue;
         double candidate = NormalizeDouble(tick.bid - trail, digits);
         if(candidate > cur_sl + min_step)       // meaningful improvement only
            new_sl = candidate;
      }
      else
      {
         if(g_pos.PriceOpen() - tick.ask < act) continue;
         double candidate = NormalizeDouble(tick.ask + trail, digits);
         if(candidate < cur_sl - min_step || cur_sl == 0)
            new_sl = candidate;
      }

      if(new_sl != cur_sl)
         g_trade.PositionModify(g_pos.Ticket(), new_sl, cur_tp);
   }
}
//+------------------------------------------------------------------+
