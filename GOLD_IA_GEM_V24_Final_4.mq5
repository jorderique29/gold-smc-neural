//+------------------------------------------------------------------+
//|  GOLD_IA_GEM_V24_Final.mq5                                       |
//|  LSTM Dual-Core + SMC Filters  |  XAUUSD M5  |  ADNX10          |
//|  v24.50 — V23 validado + modelo SELL + filtros SMC               |
//|  BUY:  LSTM 5 features [1,100,5]  — V23 validado PF=1.30         |
//|  SELL: BiLSTM 7 features [1,100,7] — nuevo modelo                |
//|  Filtros SMC: H1 bias + ADX + Kill Zone                          |
//+------------------------------------------------------------------+
#property copyright "EA V24 Final — LSTM+SMC"
#property version   "24.50"
#property strict

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\PositionInfo.mqh>

#resource "\\Files\\gold_lstm_model.onnx"      as uchar ExtModelBuy[]
#resource "\\Files\\gold_lstm_model_sell.onnx" as uchar ExtModelSell[]

CTrade        trade;
CSymbolInfo   symbolInfo;
CPositionInfo posInfo;

//+------------------------------------------------------------------+
//| PARAMETROS                                                        |
//+------------------------------------------------------------------+

input group "=== MODELOS LSTM ==="
input bool   InpEnableBuys      = true;   // Operar longs (modelo BUY V23)
input bool   InpEnableSells     = true;   // Operar shorts (modelo SELL nuevo)
input double InpMinProbBuy      = 0.35;   // Umbral BUY (V23 validado)
input double InpMinProbSell     = 0.35;   // Umbral SELL
input int    InpTimeSteps       = 100;    // Ventana temporal (velas M5)
input int    InpCooldownMinutes = 45;     // Cooldown entre trades (V23)

input group "=== FILTROS SMC ==="
input bool   InpUseH1Bias       = true;   // Filtro bias H1 (EMA20>EMA50)
input bool   InpUseADX          = true;   // Filtro ADX (evita rangos)
input int    InpADX_Period      = 14;     // Periodo ADX H1
input double InpADX_Min         = 20.0;   // ADX minimo para operar
input bool   InpUseSessionFilter= true;   // Solo Kill Zones

input group "=== SESIONES UTC ==="
input int    InpLondonStart     = 7;
input int    InpLondonEnd       = 9;
input int    InpNYStart         = 12;
input int    InpNYEnd           = 15;

input group "=== FILTROS MERCADO ==="
input int    InpMaxSpread       = 180;
input double InpMinATR_Pct      = 0.05;
input double InpMaxATR_Pct      = 0.40;
input bool   InpAvoidFridayClose= true;

input group "=== RIESGO ADNX10 ==="
input double InpRiskPercent     = 0.75;   // % balance por trade
input double InpSL_ATR_Mult     = 1.3;    // SL = 1.3x ATR M15 (V23)
input double InpTP_ATR_Mult     = 3.5;    // TP = 3.5x ATR M15 (V23)
input double InpMaxDailyLossPct = 2.5;    // Max perdida diaria %
input double InpMaxTotalDD      = 8.0;    // DD maximo % (ADNX10 limite=10%)
input int    InpATR_Period      = 14;
input int    InpMaxConsecLosses = 2;      // Circuit breaker (V23)

input group "=== GESTION POSICION V23 ==="
input bool   InpUsePeakLock         = true;
input double InpPeakLockDrawdownPct = 2.0;
input bool   InpSmartPeakLock       = true;
input bool   InpUsePartialClose     = true;
input double InpPartialClosePct     = 50.0;
input double InpBE_TriggerATR       = 1.0;
input double InpBE_SecureATR        = 0.1;
input double InpSmartLockTriggerATR = 1.5;
input double InpSmartLockSecureATR  = 0.5;

input group "=== CONFIG ==="
input ulong  InpMagic           = 240050;
input string InpComment         = "V24_LSTM_SMC";
input bool   InpDebugMode       = false;

//+------------------------------------------------------------------+
//| GLOBALES                                                          |
//+------------------------------------------------------------------+

long  g_model_buy  = INVALID_HANDLE;
long  g_model_sell = INVALID_HANDLE;

int   g_atr_m15    = INVALID_HANDLE;
int   g_ema20_h1   = INVALID_HANDLE;
int   g_ema50_h1   = INVALID_HANDLE;
int   g_ema200_h1  = INVALID_HANDLE;  // filtro macro tendencia
int   g_adx_h1     = INVALID_HANDLE;
int   g_rsi_m5     = INVALID_HANDLE;

datetime g_last_bar      = 0;
datetime g_last_trade    = 0;
int      g_cur_day       = -1;
double   g_day_bal       = 0;
double   g_peak_eq       = 0;
bool     g_halted        = false;
string   g_halt_reason   = "";

//+------------------------------------------------------------------+
//| OnInit                                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(30);
   trade.SetTypeFilling(ORDER_FILLING_IOC);

   // Cargar modelo BUY [1, 100, 5]
   if(InpEnableBuys)
   {
      g_model_buy = OnnxCreateFromBuffer(ExtModelBuy, ONNX_DEFAULT);
      if(g_model_buy == INVALID_HANDLE)
         { Print("ERROR modelo BUY: ", GetLastError()); return INIT_FAILED; }
      ulong sh_buy[] = {1, (ulong)InpTimeSteps, 5};
      if(!OnnxSetInputShape(g_model_buy, 0, sh_buy))
         { Print("ERR InputShape BUY: ", GetLastError()); return INIT_FAILED; }
      // Probar diferentes shapes de output hasta encontrar el correcto
      ulong sh1[]  = {1};
      ulong sh11[] = {1, 1};
      ulong sh12[] = {1, 2};
      bool out_ok = OnnxSetOutputShape(g_model_buy, 0, sh1);
      if(!out_ok) out_ok = OnnxSetOutputShape(g_model_buy, 0, sh11);
      if(!out_ok) out_ok = OnnxSetOutputShape(g_model_buy, 0, sh12);
      if(!out_ok)
         Print("WARN OutputShape BUY — usando shape automatico");
      Print("Modelo BUY OK | input=[1,",InpTimeSteps,",5]");
   }

   // Cargar modelo SELL [1, 100, 7]
   if(InpEnableSells)
   {
      g_model_sell = OnnxCreateFromBuffer(ExtModelSell, ONNX_DEFAULT);
      if(g_model_sell == INVALID_HANDLE)
         { Print("ERROR modelo SELL: ", GetLastError()); return INIT_FAILED; }
      ulong sh_sell[] = {1, (ulong)InpTimeSteps, 7};
      if(!OnnxSetInputShape(g_model_sell, 0, sh_sell))
         { Print("ERR InputShape SELL: ", GetLastError()); return INIT_FAILED; }
      ulong sh1s[]  = {1};
      ulong sh11s[] = {1, 1};
      ulong sh12s[] = {1, 2};
      bool out_ok = OnnxSetOutputShape(g_model_sell, 0, sh1s);
      if(!out_ok) out_ok = OnnxSetOutputShape(g_model_sell, 0, sh11s);
      if(!out_ok) out_ok = OnnxSetOutputShape(g_model_sell, 0, sh12s);
      if(!out_ok)
         Print("WARN OutputShape SELL — usando shape automatico");
      Print("Modelo SELL OK | input=[1,",InpTimeSteps,",7]");
   }

   // Indicadores
   g_atr_m15   = iATR(_Symbol, PERIOD_M15, InpATR_Period);
   g_ema20_h1  = iMA (_Symbol, PERIOD_H1,  20,  0, MODE_EMA, PRICE_CLOSE);
   g_ema50_h1  = iMA (_Symbol, PERIOD_H1,  50,  0, MODE_EMA, PRICE_CLOSE);
   g_ema200_h1 = iMA (_Symbol, PERIOD_H1,  200, 0, MODE_EMA, PRICE_CLOSE);
   g_adx_h1    = iADX(_Symbol, PERIOD_H1,  InpADX_Period);
   g_rsi_m5    = iRSI(_Symbol, PERIOD_M5,  14, PRICE_CLOSE);

   if(g_atr_m15==INVALID_HANDLE  || g_ema20_h1==INVALID_HANDLE  ||
      g_ema50_h1==INVALID_HANDLE  || g_ema200_h1==INVALID_HANDLE ||
      g_adx_h1==INVALID_HANDLE    || g_rsi_m5==INVALID_HANDLE)
      { Print("ERROR handles indicadores"); return INIT_FAILED; }

   g_day_bal = AccountInfoDouble(ACCOUNT_BALANCE);
   g_peak_eq = g_day_bal;

   Print("V24.50 LSTM+SMC | Risk=",InpRiskPercent,"% | SL=",InpSL_ATR_Mult,
         "xATR | TP=",InpTP_ATR_Mult,"xATR | MaxDD=",InpMaxTotalDD,"%");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| OnDeinit                                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_model_buy !=INVALID_HANDLE) OnnxRelease(g_model_buy);
   if(g_model_sell!=INVALID_HANDLE) OnnxRelease(g_model_sell);
   int h[]={g_atr_m15,g_ema20_h1,g_ema50_h1,g_ema200_h1,g_adx_h1,g_rsi_m5};
   for(int i=0;i<6;i++) if(h[i]!=INVALID_HANDLE) IndicatorRelease(h[i]);
}

//+------------------------------------------------------------------+
//| OnTick                                                            |
//+------------------------------------------------------------------+
void OnTick()
{
   double equity  = AccountInfoDouble(ACCOUNT_EQUITY);
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);

   // Reset diario
   MqlDateTime dt; TimeToStruct(TimeCurrent(),dt);
   if(dt.day_of_year != g_cur_day)
   {
      g_cur_day    = dt.day_of_year;
      g_day_bal    = balance;
      g_peak_eq    = balance;
      g_halted     = false;
      g_halt_reason= "";
      Print("Nuevo dia | Balance: ",g_day_bal);
   }
   if(equity > g_peak_eq) g_peak_eq = equity;

   // Gestionar posiciones (siempre)
   if(PositionsTotal() > 0) ManageExits();

   // Proteccion DD total
   if(balance>0 && (balance-equity)/balance*100 >= InpMaxTotalDD)
   { g_halted=true; g_halt_reason="MAX DD TOTAL"; }

   // Proteccion perdida diaria
   if(!g_halted && g_day_bal>0 &&
      (g_day_bal-equity)/g_day_bal*100 >= InpMaxDailyLossPct)
   { CloseAll("Max Daily Loss"); g_halted=true; g_halt_reason="MAX DAILY LOSS"; }

   // Smart Peak Lock (V23)
   if(!g_halted && InpUsePeakLock && g_peak_eq>g_day_bal)
   {
      double dd=(g_peak_eq-equity)/g_peak_eq*100.0;
      if(dd>=InpPeakLockDrawdownPct)
      {
         bool safe=true;
         if(InpSmartPeakLock)
            for(int i=PositionsTotal()-1;i>=0;i--)
            {
               if(!PositionSelectByTicket(PositionGetTicket(i))) continue;
               if(PositionGetString(POSITION_SYMBOL)!=_Symbol) continue;
               if(PositionGetInteger(POSITION_MAGIC)!=(long)InpMagic) continue;
               double op=PositionGetDouble(POSITION_PRICE_OPEN);
               double sl=PositionGetDouble(POSITION_SL);
               int pt=(int)PositionGetInteger(POSITION_TYPE);
               if(!((pt==POSITION_TYPE_BUY&&sl>=op)||(pt==POSITION_TYPE_SELL&&sl>0&&sl<=op)))
               { safe=false; break; }
            }
         if(!InpSmartPeakLock||!safe){ CloseAll("Peak Lock"); g_halted=true; g_halt_reason="PEAK LOCK"; }
         else { g_halted=true; g_halt_reason="PEAK LOCK runner libre"; }
      }
   }

   // Circuit Breaker (V23)
   if(!g_halted)
   {
      MqlDateTime ds; TimeToStruct(TimeCurrent(),ds);
      ds.hour=0;ds.min=0;ds.sec=0;
      if(ConsecLosses(StructToTime(ds))>=InpMaxConsecLosses)
      { g_halted=true; g_halt_reason="CIRCUIT BREAKER"; Print("CIRCUIT BREAKER activado"); }
   }

   if(g_halted) return;

   // Viernes tarde
   if(InpAvoidFridayClose && dt.day_of_week==5 && dt.hour>=20) return;

   // Nueva vela M5
   datetime cur=iTime(_Symbol,PERIOD_M5,0);
   if(cur==g_last_bar) return;
   g_last_bar=cur;

   // Spread
   symbolInfo.RefreshRates();
   if(symbolInfo.Spread()>InpMaxSpread) return;

   // ATR M15
   double atr=ObtenerATR();
   if(atr<=0) return;
   double atr_pct=(symbolInfo.Bid()>0)?atr/symbolInfo.Bid()*100:0;
   if(atr_pct<InpMinATR_Pct||atr_pct>InpMaxATR_Pct) return;

   // Cooldown
   if(TimeCurrent()-g_last_trade<(datetime)(InpCooldownMinutes*60)) return;

   // Sin posicion abierta
   if(TienePos()) return;

   // Kill Zone
   if(InpUseSessionFilter&&!EnKillZone()) return;

   // ── FILTRO SMC 1: BIAS H1 (EMA20 vs EMA50) ──
   int bias_h1 = 0;
   if(InpUseH1Bias)
   {
      double e20[],e50[],cl[];
      ArraySetAsSeries(e20,true); ArraySetAsSeries(e50,true); ArraySetAsSeries(cl,true);
      if(CopyBuffer(g_ema20_h1,0,1,1,e20)<1) return;
      if(CopyBuffer(g_ema50_h1,0,1,1,e50)<1) return;
      if(CopyClose(_Symbol,PERIOD_H1,1,1,cl)<1) return;
      if(e20[0]>e50[0]&&cl[0]>e50[0])      bias_h1 =  1;
      else if(e20[0]<e50[0]&&cl[0]<e50[0]) bias_h1 = -1;
      if(bias_h1==0) return;
   }

   // ── FILTRO MACRO: EMA200 H1 ──
   // Solo longs si precio > EMA200, solo shorts si precio < EMA200
   // Evita operar contra la tendencia de largo plazo
   {
      double e200[],cl200[];
      ArraySetAsSeries(e200,true); ArraySetAsSeries(cl200,true);
      if(CopyBuffer(g_ema200_h1,0,1,1,e200)>=1 &&
         CopyClose(_Symbol,PERIOD_H1,1,1,cl200)>=1)
      {
         bool precio_sobre_ema200 = (cl200[0] > e200[0]);
         // Bloquear longs si precio bajo EMA200
         if(!precio_sobre_ema200 && InpEnableBuys  && bias_h1==1)  bias_h1=0;
         // Bloquear shorts si precio sobre EMA200
         if( precio_sobre_ema200 && InpEnableSells && bias_h1==-1) bias_h1=0;
         if(bias_h1==0)
         {
            if(InpDebugMode) Print("EMA200 filtro — precio ",
               precio_sobre_ema200?"SOBRE":"BAJO"," EMA200, señal bloqueada");
            return;
         }
      }
   }

   // ── FILTRO SMC 2: ADX ──
   if(InpUseADX)
   {
      double adx[]; ArraySetAsSeries(adx,true);
      if(CopyBuffer(g_adx_h1,0,1,1,adx)>=1)
         if(adx[0]<InpADX_Min)
         {
            if(InpDebugMode) Print("ADX=",DoubleToString(adx[0],1)," < ",InpADX_Min," — rango lateral");
            return;
         }
   }

   if(InpDebugMode) Print("Filtros SMC OK | H1=",bias_h1);

   // ── MODELO BUY (long) — solo si H1 es bull o sin filtro H1 ──
   if(InpEnableBuys && (bias_h1==1 || !InpUseH1Bias))
   {
      float tensor_buy[];
      if(PrepareTensorBuy(tensor_buy))
      {
         float prob_buy = RunBuy(tensor_buy);
         if(InpDebugMode) Print("BUY prob=",DoubleToString(prob_buy,4));
         if(prob_buy >= (float)InpMinProbBuy)
         {
            Print("LONG | prob=",DoubleToString(prob_buy,3)," | H1=",bias_h1);
            EjecutarTrade(ORDER_TYPE_BUY, atr, (double)prob_buy);
            return;
         }
      }
   }

   // ── MODELO SELL (short) — solo si H1 es bear o sin filtro H1 ──
   if(InpEnableSells && (bias_h1==-1 || !InpUseH1Bias))
   {
      float tensor_sell[];
      if(PrepareTensorSell(tensor_sell))
      {
         float prob_sell = RunSell(tensor_sell);
         if(InpDebugMode) Print("SELL prob=",DoubleToString(prob_sell,4));
         if(prob_sell >= (float)InpMinProbSell)
         {
            Print("SHORT | prob=",DoubleToString(prob_sell,3)," | H1=",bias_h1);
            EjecutarTrade(ORDER_TYPE_SELL, atr, (double)prob_sell);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| PREPARAR TENSOR BUY — [1, TimeSteps, 5] (OHLCV normalizado)     |
//+------------------------------------------------------------------+
bool PrepareTensorBuy(float &tensor[])
{
   MqlRates rates[];
   ArraySetAsSeries(rates,true);
   if(CopyRates(_Symbol,PERIOD_M5,1,InpTimeSteps,rates)!=InpTimeSteps) return false;

   double mn_p=DBL_MAX,mx_p=-DBL_MAX,mn_v=DBL_MAX,mx_v=-DBL_MAX;
   for(int i=0;i<InpTimeSteps;i++)
   {
      if(rates[i].low   < mn_p) mn_p=rates[i].low;
      if(rates[i].high  > mx_p) mx_p=rates[i].high;
      if((double)rates[i].tick_volume < mn_v) mn_v=(double)rates[i].tick_volume;
      if((double)rates[i].tick_volume > mx_v) mx_v=(double)rates[i].tick_volume;
   }

   ArrayResize(tensor, InpTimeSteps*5);
   int idx=0;
   for(int i=InpTimeSteps-1;i>=0;i--)
   {
      tensor[idx++]=NormVal(rates[i].open,               mn_p,mx_p);
      tensor[idx++]=NormVal(rates[i].high,               mn_p,mx_p);
      tensor[idx++]=NormVal(rates[i].low,                mn_p,mx_p);
      tensor[idx++]=NormVal(rates[i].close,              mn_p,mx_p);
      tensor[idx++]=NormVal((double)rates[i].tick_volume,mn_v,mx_v);
   }
   return true;
}

//+------------------------------------------------------------------+
//| PREPARAR TENSOR SELL — [1, TimeSteps, 7] (OHLCV+RSI+Slope)     |
//+------------------------------------------------------------------+
bool PrepareTensorSell(float &tensor[])
{
   MqlRates rates[];
   ArraySetAsSeries(rates,true);
   if(CopyRates(_Symbol,PERIOD_M5,1,InpTimeSteps,rates)!=InpTimeSteps) return false;

   double rsi_buf[];
   ArraySetAsSeries(rsi_buf,true);
   bool has_rsi=(g_rsi_m5!=INVALID_HANDLE &&
                 CopyBuffer(g_rsi_m5,0,1,InpTimeSteps,rsi_buf)==InpTimeSteps);
   if(!has_rsi){ ArrayResize(rsi_buf,InpTimeSteps); ArrayFill(rsi_buf,0,InpTimeSteps,50.0); }

   double mn_p=DBL_MAX,mx_p=-DBL_MAX,mn_v=DBL_MAX,mx_v=-DBL_MAX;
   for(int i=0;i<InpTimeSteps;i++)
   {
      if(rates[i].low   < mn_p) mn_p=rates[i].low;
      if(rates[i].high  > mx_p) mx_p=rates[i].high;
      if((double)rates[i].tick_volume < mn_v) mn_v=(double)rates[i].tick_volume;
      if((double)rates[i].tick_volume > mx_v) mx_v=(double)rates[i].tick_volume;
   }

   ArrayResize(tensor, InpTimeSteps*7);
   int idx=0;
   for(int i=InpTimeSteps-1;i>=0;i--)
   {
      tensor[idx++]=NormVal(rates[i].open,               mn_p,mx_p);
      tensor[idx++]=NormVal(rates[i].high,               mn_p,mx_p);
      tensor[idx++]=NormVal(rates[i].low,                mn_p,mx_p);
      tensor[idx++]=NormVal(rates[i].close,              mn_p,mx_p);
      tensor[idx++]=NormVal((double)rates[i].tick_volume,mn_v,mx_v);
      // RSI normalizado [0,1]
      tensor[idx++]=(float)(rsi_buf[i]/100.0);
      // Slope: pendiente 5 velas, clip ±0.003, norm [-1,1]
      double slope=0;
      if(i+5<InpTimeSteps)
         slope=(rates[i].close-rates[i+5].close)/(rates[i].close+1e-10);
      tensor[idx++]=(float)MathMax(-1.0,MathMin(1.0,slope/0.003));
   }
   return true;
}

//+------------------------------------------------------------------+
//| INFERENCIA                                                        |
//+------------------------------------------------------------------+
float RunBuy(const float &tensor[])
{
   float out[1]={0.0f};
   if(g_model_buy==INVALID_HANDLE) return 0.0f;
   if(!OnnxRun(g_model_buy,ONNX_DEFAULT,tensor,out)) return 0.0f;
   return out[0];
}

float RunSell(const float &tensor[])
{
   float out[1]={0.0f};
   if(g_model_sell==INVALID_HANDLE) return 0.0f;
   if(!OnnxRun(g_model_sell,ONNX_DEFAULT,tensor,out)) return 0.0f;
   return out[0];
}

//+------------------------------------------------------------------+
//| EJECUTAR TRADE                                                    |
//+------------------------------------------------------------------+
void EjecutarTrade(ENUM_ORDER_TYPE tipo, double atr, double prob)
{
   symbolInfo.RefreshRates();
   double price=(tipo==ORDER_TYPE_BUY)?symbolInfo.Ask():symbolInfo.Bid();
   int digs=_Digits;

   double sl_d = atr*InpSL_ATR_Mult;
   double tp_d = atr*InpTP_ATR_Mult;
   double min_sl=200*symbolInfo.Point();
   if(sl_d<min_sl) sl_d=min_sl;

   double sl=NormalizeDouble((tipo==ORDER_TYPE_BUY)?price-sl_d:price+sl_d,digs);
   double tp=NormalizeDouble((tipo==ORDER_TYPE_BUY)?price+tp_d:price-tp_d,digs);

   double bal =AccountInfoDouble(ACCOUNT_BALANCE);
   double risk=bal*InpRiskPercent/100.0;
   double tv  =SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_VALUE);
   double ts  =SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);
   double min_l=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN);
   double max_l=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MAX);
   double step =SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
   if(tv<=0||ts<=0) return;
   double lots=risk/(sl_d/ts*tv);
   lots=MathFloor(lots/step)*step;
   lots=MathMax(min_l,MathMin(max_l,lots));

   if(lots==min_l&&InpUsePartialClose)
   {
      double mg; double t2=min_l*2;
      if(OrderCalcMargin(tipo,_Symbol,t2,price,mg)&&
         mg<=AccountInfoDouble(ACCOUNT_MARGIN_FREE)*0.8) lots=t2;
   }

   string cmt=StringFormat("%s|p=%.2f",InpComment,prob);
   bool ok=(tipo==ORDER_TYPE_BUY)
           ?trade.Buy (lots,_Symbol,price,sl,tp,cmt)
           :trade.Sell(lots,_Symbol,price,sl,tp,cmt);

   if(ok)
   {
      g_last_trade=TimeCurrent();
      Print((tipo==ORDER_TYPE_BUY?"LONG":"SHORT"),
            " lot=",lots," sl=",sl," tp=",tp," atr=",DoubleToString(atr,2));
   }
   else Print("Trade err:",trade.ResultRetcode()," ",trade.ResultRetcodeDescription());
}

//+------------------------------------------------------------------+
//| GESTION DE SALIDAS — Shadow Trailing V23                         |
//+------------------------------------------------------------------+
void ManageExits()
{
   double atr=ObtenerATR();
   if(atr<=0) return;

   double min_stop=MathMax(
      (double)SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL),
      (double)SymbolInfoInteger(_Symbol,SYMBOL_TRADE_FREEZE_LEVEL)
   )*symbolInfo.Point()+5*symbolInfo.Point();

   double step_vol=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
   double min_lot =SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN);
   double be_trig =atr*InpBE_TriggerATR;
   double be_sec  =atr*InpBE_SecureATR;
   double lk_trig =atr*InpSmartLockTriggerATR;
   double lk_sec  =atr*InpSmartLockSecureATR;

   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong tk=PositionGetTicket(i);
      if(!PositionSelectByTicket(tk)) continue;
      if(PositionGetString(POSITION_SYMBOL)!=_Symbol) continue;
      if(PositionGetInteger(POSITION_MAGIC)!=(long)InpMagic) continue;

      int    pt  =(int)PositionGetInteger(POSITION_TYPE);
      double pop =PositionGetDouble(POSITION_PRICE_OPEN);
      double psl =PositionGetDouble(POSITION_SL);
      double ptp =PositionGetDouble(POSITION_TP);
      double pvol=PositionGetDouble(POSITION_VOLUME);
      double bid =symbolInfo.Bid();
      double ask =symbolInfo.Ask();
      double pcur=(pt==POSITION_TYPE_BUY)?bid:ask;
      double prof=(pt==POSITION_TYPE_BUY)?pcur-pop:pop-pcur;
      bool   pdone=IsPartialDone(tk);

      if(pt==POSITION_TYPE_BUY)
      {
         if(InpUsePartialClose&&!pdone&&prof>=atr*1.5)
         {
            double cl=MathFloor(pvol*(InpPartialClosePct/100.0)/step_vol)*step_vol;
            if(cl>=min_lot&&cl<pvol) trade.PositionClosePartial(tk,cl);
         }
         double nbe=NormalizeDouble(pop+be_sec,_Digits);
         if(prof>=be_trig&&psl<pop&&nbe>psl&&nbe+min_stop<=pcur)
            trade.PositionModify(tk,nbe,ptp);
         double tr=NormalizeDouble(pcur-lk_sec,_Digits);
         if(prof>=lk_trig&&psl>=pop&&tr>psl&&tr+min_stop<=pcur)
            trade.PositionModify(tk,tr,ptp);
      }
      else
      {
         if(InpUsePartialClose&&!pdone&&prof>=atr*1.5)
         {
            double cl=MathFloor(pvol*(InpPartialClosePct/100.0)/step_vol)*step_vol;
            if(cl>=min_lot&&cl<pvol) trade.PositionClosePartial(tk,cl);
         }
         double nbe=NormalizeDouble(pop-be_sec,_Digits);
         if(prof>=be_trig&&psl>pop&&nbe<psl&&nbe-min_stop>=pcur)
            trade.PositionModify(tk,nbe,ptp);
         double tr=NormalizeDouble(pcur+lk_sec,_Digits);
         if(prof>=lk_trig&&psl<=pop&&tr<psl&&tr-min_stop>=pcur)
            trade.PositionModify(tk,tr,ptp);
      }
   }
}

//+------------------------------------------------------------------+
//| UTILIDADES                                                        |
//+------------------------------------------------------------------+

float NormVal(double v, double mn, double mx)
{
   if(mx<=mn) return 0.0f;
   return (float)((v-mn)/(mx-mn));
}

double ObtenerATR()
{
   double b[]; ArraySetAsSeries(b,true);
   if(CopyBuffer(g_atr_m15,0,1,1,b)<1) return 0;
   return b[0];
}

bool EnKillZone()
{
   int h=(int)((TimeCurrent()%86400)/3600);
   return((h>=InpLondonStart&&h<InpLondonEnd)||(h>=InpNYStart&&h<InpNYEnd));
}

bool TienePos()
{
   for(int i=0;i<PositionsTotal();i++)
   {
      ulong tk=PositionGetTicket(i);
      if(PositionGetString(POSITION_SYMBOL)==_Symbol&&
         PositionGetInteger(POSITION_MAGIC)==(long)InpMagic) return true;
   }
   return false;
}

int ConsecLosses(datetime day_start)
{
   if(!HistorySelect(day_start,TimeCurrent())) return 0;
   int c=0,mx=0;
   for(int i=0;i<HistoryDealsTotal();i++)
   {
      ulong tk=HistoryDealGetTicket(i);
      if(HistoryDealGetString(tk,DEAL_SYMBOL)!=_Symbol) continue;
      if(HistoryDealGetInteger(tk,DEAL_MAGIC)!=(long)InpMagic) continue;
      if(HistoryDealGetInteger(tk,DEAL_ENTRY)!=DEAL_ENTRY_OUT) continue;
      if(HistoryDealGetDouble(tk,DEAL_PROFIT)<0){c++;if(c>mx)mx=c;}
      else c=0;
   }
   return mx;
}

bool IsPartialDone(ulong pos_id)
{
   if(!HistorySelectByPosition(pos_id)) return false;
   int out=0;
   for(int i=0;i<HistoryDealsTotal();i++)
      if(HistoryDealGetInteger(HistoryDealGetTicket(i),DEAL_ENTRY)==DEAL_ENTRY_OUT) out++;
   return(out>0);
}

void CloseAll(string reason)
{
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong tk=PositionGetTicket(i);
      if(PositionGetString(POSITION_SYMBOL)==_Symbol&&
         PositionGetInteger(POSITION_MAGIC)==(long)InpMagic)
         trade.PositionClose(tk);
   }
   Print("Cerradas posiciones: ",reason);
}

//+------------------------------------------------------------------+
//| OnTester — criterio de optimizacion genetica                     |
//| Maximiza: PF * WR, penaliza DD > 8% y trades < 20               |
//+------------------------------------------------------------------+
double OnTester()
{
   double trades  = TesterStatistics(STAT_TRADES);
   double pf      = TesterStatistics(STAT_PROFIT_FACTOR);
   double dd_pct  = TesterStatistics(STAT_EQUITY_DD_RELATIVE);
   double profit  = TesterStatistics(STAT_PROFIT);
   double win_t   = TesterStatistics(STAT_PROFIT_TRADES);
   double wr      = trades > 0 ? win_t / trades : 0;

   // Minimo 5 trades para tener alguna señal
   if(trades < 5) return 0.0;

   // Penalizar DD extremo
   if(dd_pct >= 10.0) return 0.0;

   // Si pierde dinero: score muy bajo pero no negativo
   // (para que el genetico pueda comparar y mejorar)
   if(profit <= 0)
      return MathMax(0.001, 0.5 / (1.0 + MathAbs(profit)));

   // Score principal: PF * sqrt(trades) * penalizacion_DD
   // sqrt(trades) premia mas trades pero sin dominar
   double dd_factor  = MathMax(0.1, 1.0 - dd_pct/15.0);
   double vol_factor = MathSqrt(MathMin(trades, 100.0)) / 10.0;
   double score      = pf * vol_factor * dd_factor;

   return score;
}
//+------------------------------------------------------------------+
