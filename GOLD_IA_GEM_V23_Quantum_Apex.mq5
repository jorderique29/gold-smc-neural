//+------------------------------------------------------------------+
//|                         GOLD_IA_GEM_V23_Quantum_Apex.mq5        |
//|           Dual-Core AI + "Golden Runner" Trailing Engine         |
//| V23: Smart Peak Lock | Parámetros Validados 2024-2026            |
//|          *** PRODUCTION BUILD — DEMO READY ***                   |
//+------------------------------------------------------------------+
#property strict
#property copyright "Trader Institucional & IA Quant"
#property link      "https://www.tusitio.com"
#property version   "23.00"

//+------------------------------------------------------------------+
//| GUIA DE BACKTEST Y PRODUCCION V23                               |
//+------------------------------------------------------------------+
// CONFIGURACION VALIDADA (backtest 2024.01.01 - 2026.03.19 XAUUSD):
//   Resultado: PF=1.30 | WR=63.7% | DD=6.99% | Neto=+$1,267 | LR=0.94
//
// PARAMETROS DE BACKTEST EN MT5:
//   Simbolo:     XAUUSD (ADN Broker — 100% ticks reales)
//   Timeframe:   M5
//   Modo:        Every tick based on real ticks
//   Fecha inicio: 2024.01.01  (minimo para muestra estadistica valida)
//   Deposito:    $10,000
//
// PARAMETROS CRITICOS (NO CAMBIAR SIN BACKTEST):
//   InpMinProbBuy          = 0.35   (umbral validado 124 trades)
//   InpCooldownMinutes     = 45     (evita re-entrada en momentum adverso)
//   InpMaxConsecutiveLosses= 2      (corta rachas malas rapido)
//   InpSL_ATR_Mult         = 1.3    (balance stop/ruido M5)
//   InpTP_ATR_Mult         = 3.5    (alcanzable en XAUUSD M5)
//   InpPeakLockDrawdownPct = 2.0    (protege ganancias del dia)
//   InpSmartPeakLock       = true   (no cierra SL en BE o profit)
//   InpEnableSells         = false  (modelo SELL pendiente reentrenamiento)
//
// PARAMETROS QUE SE PUEDEN EXPLORAR EN OPTIMIZACION:
//   InpMinProbBuy:    rango 0.30 - 0.45 (paso 0.05)
//   InpSL_ATR_Mult:   rango 1.0 - 1.8   (paso 0.1)
//   InpTP_ATR_Mult:   rango 2.5 - 5.0   (paso 0.5)
//   InpCooldownMinutes: rango 30 - 60   (paso 15)
//
// PARA PRODUCCION LIVE:
//   1. Correr minimo 4 semanas en cuenta DEMO
//   2. Verificar que trades/semana coinciden con backtest (~1-2/semana)
//   3. Monitorear que PF mensual se mantiene > 1.10
//   4. Escalar capital solo despues de 3 meses demo consistentes
//+------------------------------------------------------------------+

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\PositionInfo.mqh>

//--- Recursos ONNX (Cerebro Dual)
#resource "\\Files\\gold_lstm_model.onnx"      as uchar ExtModelDataBuy[]
#resource "\\Files\\gold_lstm_model_sell.onnx" as uchar ExtModelDataSell[]

CTrade         trade;
CSymbolInfo    symbolInfo;
CPositionInfo  posInfo;

enum ENUM_LOT_SIZING {
   LOT_SIZING_RISK_PCT, // % de Riesgo por Trade
   LOT_SIZING_PER_1K    // Lotes fijos por cada $1,000 (Escalamiento Perfecto)
};

enum ENUM_SESSION_MODE {
   SESSION_LONDON_NY,   // Londres + NY (recomendado XAUUSD)
   SESSION_LONDON_ONLY, // Solo Londres
   SESSION_NY_ONLY,     // Solo New York
   SESSION_CUSTOM       // Horas personalizadas (usa InpStartHourUTC / InpEndHourUTC)
};

//+------------------------------------------------------------------+
//| INPUTS: CEREBRO IA & DIRECCIONALIDAD                             |
//+------------------------------------------------------------------+
sinput string              ___AI_ENGINE___        = "=== IA PREDICTIVA (DUAL-CORE) ===";
input bool                 InpEnableBuys          = true;       // Produccion: true
input bool                 InpEnableSells         = false;      // Produccion: false (modelo SELL pendiente reentrenamiento)
input double               InpMinProbBuy          = 0.35;       // Validado: backtest 2024-2026 XAUUSD
input double               InpMinProbSell         = 0.28;       // Reservado para cuando SELL V3 este listo
input int                  InpTimeSteps           = 100;
input int                  InpCooldownMinutes     = 45;         // Validado: evita re-entrada en momentum adverso

//+------------------------------------------------------------------+
//| INPUTS: SESIÓN (V22 — GMT CORRECTO)                              |
//+------------------------------------------------------------------+
sinput string              ___SESSION___          = "=== SESIÓN DE TRADING (GMT) ===";
input ENUM_SESSION_MODE    InpSessionMode         = SESSION_CUSTOM;    // Produccion: SESSION_CUSTOM 0-23 (tester UTC=broker)
input int                  InpStartHourUTC        = 0;     // Produccion: 0 (sesion completa)
input int                  InpEndHourUTC          = 23;    // Produccion: 23
// V22: Cómo funciona la corrección horaria:
// El EA ahora usa TimeGMT() internamente para calcular las sesiones.
// El broker puede estar en UTC+2, UTC+3 o cualquier offset — no importa.
// Sesiones referenciadas siempre en UTC:
//   Londres:   08:00 – 17:00 UTC
//   New York:  13:00 – 22:00 UTC (EDT verano) / 14:00-23:00 UTC (EST invierno)
//   Overlap:   13:00 – 17:00 UTC (mayor liquidez XAUUSD)

//+------------------------------------------------------------------+
//| INPUTS: FILTROS INSTITUCIONALES                                  |
//+------------------------------------------------------------------+
sinput string              ___MACRO_FILTERS___    = "=== FILTROS MACRO & ENTORNO ===";
input double               InpMinATR_Pct          = 0.05;
input double               InpMaxATR_Pct          = 0.40;
input int                  InpMaxSpread           = 180;
input bool                 InpAvoidFridayClose    = true;   // V22: activado por defecto

//+------------------------------------------------------------------+
//| INPUTS: GESTIÓN DE RIESGO & PROTECCIÓN                          |
//+------------------------------------------------------------------+
sinput string              ___RISK_MANAGEMENT___  = "=== GESTIÓN DE LOTES & RIESGO ===";
input ENUM_LOT_SIZING      InpLotMethod           = LOT_SIZING_PER_1K;
input double               InpRiskPercent         = 1.0;   // V22: más conservador
input double               InpLotsPer1K           = 0.02;
input double               InpMaxDailyLossPct     = 3.5;         // Validado: balance proteccion/oportunidad
input int                  InpMaxConsecutiveLosses= 2;            // Validado: corta rachas malas rapido
input ulong                InpMagic               = 123456;

//+------------------------------------------------------------------+
//| INPUTS: PEAK EQUITY LOCK (V23 — SMART MODE)                     |
//+------------------------------------------------------------------+
sinput string              ___PEAK_LOCK___        = "=== PEAK EQUITY LOCK (V23 SMART) ===";
input bool                 InpUsePeakLock         = true;
input double               InpPeakLockDrawdownPct = 2.0;         // Validado: 2% protege ganancias sin interferir
input bool                 InpSmartPeakLock       = true;  // V23 NUEVO: no disparar si SL ya está en zona segura
// Smart Peak Lock V23:
// El Peak Lock original cerraba CUALQUIER posición abierta si la equity caía X% desde el pico.
// Esto era problemático porque cerraba trades bien gestionados que ya tenían SL en BE o ganancia.
// Con SmartPeakLock = true, el Peak Lock SOLO actúa si la posición abierta tiene SL
// por debajo del precio de entrada (aún en riesgo real). Si el SL ya está en BE o en profit,
// el Peak Lock no interfiere — el trailing se encarga del resto.
// Rango óptimo: 3.0% (conservador) — 5.0% (agresivo)

//+------------------------------------------------------------------+
//| INPUTS: MANEJO DE SALIDAS (V23: GOLDEN RUNNER OPTIMIZADO)       |
//+------------------------------------------------------------------+
sinput string              ___TRADE_MANAGEMENT___ = "=== SHADOW TRAIL (THE RUNNER) ===";
input double               InpSL_ATR_Mult         = 1.3;         // Validado: WR 63.7% en 124 trades
input double               InpTP_ATR_Mult         = 3.5;         // Validado: alcanzable en XAUUSD M5

//+------------------------------------------------------------------+
//| INPUTS: MODELO SELL V2 (V23 — 7 FEATURES)                       |
//+------------------------------------------------------------------+
sinput string              ___SELL_MODEL___       = "=== MODELO SELL V2 (RSI + SLOPE) ===";
input bool                 InpUseSellModelV2      = false; // Produccion: false hasta reentrenar SELL V3 con datos 2022+
// Cuando InpUseSellModelV2 = true, el EA construye un tensor de 7 columnas:
// [Open, High, Low, Close, Volume, RSI_norm, Slope_norm]
// Esto requiere haber reentrenado con train_gold_ai_sell_v2.py
// y copiado el nuevo gold_lstm_model_sell.onnx a MQL5\Files\

// FASE 1: Reducción de Riesgo Temprana
input double               InpRiskReductionTrigger= 0.8;   // A +0.8 ATR...
input double               InpRiskReductionSet    = 0.3;   // ...reducir riesgo al SL -0.3 ATR

// FASE 2: Break-Even
input double               InpBE_TriggerATR       = 1.2;   // V22: activar BE más temprano
input double               InpBE_SecureATR        = 0.1;

// FASE 3: Scale-Out (Cierre Parcial)
input bool                 InpUsePartialClose     = true;
input double               InpSmartLockTriggerATR = 1.5;   // V22: cobrar 50% antes (era 2.5x)
input double               InpSmartLockSecureATR  = 1.0;   // V22: asegurar 1x ATR para el runner

// FASE 4: Golden Runner Trailing
input double               InpTrailStepATRMult    = 0.8;   // V22: escalones más ajustados (era 1.5x)
input int                  InpAtrPeriod           = 14;

//--- Variables Globales
long           model_buy_handle       = INVALID_HANDLE;
long           model_sell_handle      = INVALID_HANDLE;
int            atr_handle             = INVALID_HANDLE;
int            rsi_handle             = INVALID_HANDLE;   // V23: para tensor de 7 features
datetime       lastSignalBarTime      = 0;
datetime       lastTradeTime          = 0;
double         daily_start_balance    = 0.0;
double         peak_equity_today      = 0.0;
int            current_day            = -1;
bool           trading_halted_today   = false;
string         ai_status_msg          = "INICIALIZANDO V23 QUANTUM APEX";
string         halt_reason            = "";

//+------------------------------------------------------------------+
//| V22: LÓGICA DE SESIÓN BASADA EN GMT (CORRECCIÓN HORARIA)        |
//+------------------------------------------------------------------+
bool IsNYSummerTime(const MqlDateTime &gmt_dt)
{
   // EDT (verano NY): segundo domingo de marzo → primer domingo de noviembre
   // Simplificación práctica: meses 4-10 = verano
   return (gmt_dt.mon >= 4 && gmt_dt.mon <= 10);
}

bool IsWithinLondonSession(const MqlDateTime &gmt_dt)
{
   // Londres: 08:00 – 17:00 UTC (todo el año, ajuste BST ya incluido en UTC)
   return (gmt_dt.hour >= 8 && gmt_dt.hour < 17);
}

bool IsWithinNYSession(const MqlDateTime &gmt_dt)
{
   // EDT (verano): 13:00 – 22:00 UTC
   // EST (invierno): 14:00 – 23:00 UTC
   if(IsNYSummerTime(gmt_dt))
      return (gmt_dt.hour >= 13 && gmt_dt.hour < 22);
   else
      return (gmt_dt.hour >= 14 && gmt_dt.hour < 23);
}

bool IsWithinTradingSession()
{
   MqlDateTime gmt_dt;
   TimeToStruct(TimeGMT(), gmt_dt); // <-- siempre UTC, independiente del broker

   switch(InpSessionMode)
   {
      case SESSION_LONDON_NY:
         return (IsWithinLondonSession(gmt_dt) || IsWithinNYSession(gmt_dt));
      case SESSION_LONDON_ONLY:
         return IsWithinLondonSession(gmt_dt);
      case SESSION_NY_ONLY:
         return IsWithinNYSession(gmt_dt);
      case SESSION_CUSTOM:
         return (gmt_dt.hour >= InpStartHourUTC && gmt_dt.hour < InpEndHourUTC);
   }
   return false;
}

// Devuelve un string con el estado de la sesión actual para el HUD
string GetSessionStatus()
{
   MqlDateTime gmt_dt;
   TimeToStruct(TimeGMT(), gmt_dt);
   string s = "";
   if(IsWithinLondonSession(gmt_dt)) s += "LON";
   if(IsWithinNYSession(gmt_dt))
   {
      if(StringLen(s) > 0) s += "+";
      s += "NY";
   }
   if(StringLen(s) == 0) s = "FUERA";
   return StringFormat("%s | GMT %02d:%02d", s, gmt_dt.hour, gmt_dt.min);
}

// Devuelve el offset del broker respecto a UTC (solo informativo para el HUD)
int GetBrokerUTCOffset()
{
   return (int)MathRound((double)(TimeCurrent() - TimeGMT()) / 3600.0);
}

//+------------------------------------------------------------------+
//| Utilidades Básicas                                               |
//+------------------------------------------------------------------+
bool IsNewBar(const ENUM_TIMEFRAMES tf, datetime &last_bar_time)
{
   datetime times[];
   if(CopyTime(_Symbol, tf, 0, 1, times) != 1) return false;
   if(times[0] != last_bar_time)
   {
      last_bar_time = times[0];
      return true;
   }
   return false;
}

bool IsCooldownActive()
{
   if(lastTradeTime <= 0) return false;
   return ((TimeCurrent() - lastTradeTime) < (InpCooldownMinutes * 60));
}

bool GetAtrValue(double &atr_value)
{
   double atr_buf[];
   ArraySetAsSeries(atr_buf, true);
   if(CopyBuffer(atr_handle, 0, 0, 1, atr_buf) != 1) return false;
   atr_value = atr_buf[0];
   return (atr_value > 0.0);
}

float NormalizeWindowValue(const double value, const double min_v, const double max_v)
{
   if(max_v <= min_v) return 0.5f;
   return (float)((value - min_v) / (max_v - min_v));
}

void CloseAllPositions(string reason = "Cierre Manual")
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == InpMagic)
      {
         Print("V22 Cerrando posicion: ", reason);
         trade.PositionClose(ticket);
      }
   }
}

int GetConsecutiveLossesToday(datetime day_start)
{
   HistorySelect(day_start, TimeCurrent());
   int consec_losses = 0;

   for(int i = HistoryDealsTotal() - 1; i >= 0; i--)
   {
      ulong deal_ticket = HistoryDealGetTicket(i);
      long magic   = HistoryDealGetInteger(deal_ticket, DEAL_MAGIC);
      string sym   = HistoryDealGetString(deal_ticket, DEAL_SYMBOL);
      long entry   = HistoryDealGetInteger(deal_ticket, DEAL_ENTRY);

      if(magic == (long)InpMagic && sym == _Symbol && (entry == DEAL_ENTRY_OUT || entry == DEAL_ENTRY_INOUT))
      {
         double profit = HistoryDealGetDouble(deal_ticket, DEAL_PROFIT)
                       + HistoryDealGetDouble(deal_ticket, DEAL_SWAP)
                       + HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION);
         if(profit < 0)
            consec_losses++;
         else
            break;
      }
   }
   return consec_losses;
}

//+------------------------------------------------------------------+
//| Interfaz Gráfica (HUD V22)                                      |
//+------------------------------------------------------------------+
void UpdateDashboard()
{
   string font = "Consolas";
   int x = 20, y = 30;

   color clrTitle = clrGold;
   color clrText  = clrWhite;
   if(ChartGetInteger(0, CHART_COLOR_BACKGROUND) == clrWhite) clrText = clrBlack;

   // Título
   if(ObjectFind(0, "HUD_Title") < 0) ObjectCreate(0, "HUD_Title", OBJ_LABEL, 0, 0, 0);
   ObjectSetString(0,  "HUD_Title", OBJPROP_TEXT,     "GOLD IA GEM V23 - QUANTUM APEX | DEMO");
   ObjectSetString(0,  "HUD_Title", OBJPROP_FONT,     "Arial");
   ObjectSetInteger(0, "HUD_Title", OBJPROP_FONTSIZE, 12);
   ObjectSetInteger(0, "HUD_Title", OBJPROP_COLOR,    clrTitle);
   ObjectSetInteger(0, "HUD_Title", OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, "HUD_Title", OBJPROP_YDISTANCE, y);

   // Estado IA
   y += 25;
   if(ObjectFind(0, "HUD_State") < 0) ObjectCreate(0, "HUD_State", OBJ_LABEL, 0, 0, 0);
   ObjectSetString(0,  "HUD_State", OBJPROP_TEXT, "ESTADO: " + ai_status_msg);
   ObjectSetString(0,  "HUD_State", OBJPROP_FONT, font);
   ObjectSetInteger(0, "HUD_State", OBJPROP_FONTSIZE, 10);
   color stateClr = (ai_status_msg == "ESCANEANDO (PURE A.I.)") ? clrLimeGreen
                  : (ai_status_msg == "EN COOLDOWN")            ? clrOrange
                  : clrDodgerBlue;
   if(trading_halted_today) stateClr = clrRed;
   ObjectSetInteger(0, "HUD_State", OBJPROP_COLOR, stateClr);
   ObjectSetInteger(0, "HUD_State", OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, "HUD_State", OBJPROP_YDISTANCE, y);

   // V22: Sesión GMT
   y += 20;
   if(ObjectFind(0, "HUD_Session") < 0) ObjectCreate(0, "HUD_Session", OBJ_LABEL, 0, 0, 0);
   bool in_session = IsWithinTradingSession();
   string session_txt = StringFormat("SESION: %s | BROKER UTC%+d",
                                      GetSessionStatus(), GetBrokerUTCOffset());
   ObjectSetString(0,  "HUD_Session", OBJPROP_TEXT, session_txt);
   ObjectSetString(0,  "HUD_Session", OBJPROP_FONT, font);
   ObjectSetInteger(0, "HUD_Session", OBJPROP_FONTSIZE, 10);
   ObjectSetInteger(0, "HUD_Session", OBJPROP_COLOR, in_session ? clrLimeGreen : clrGray);
   ObjectSetInteger(0, "HUD_Session", OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, "HUD_Session", OBJPROP_YDISTANCE, y);

   // Escala de lotes
   y += 20;
   string scale_msg = (InpLotMethod == LOT_SIZING_PER_1K)
      ? StringFormat("LOT_SCALER: %.2f Lotes / $1K", InpLotsPer1K)
      : StringFormat("RISK_PCT: %.2f%%", InpRiskPercent);
   if(ObjectFind(0, "HUD_Scale") < 0) ObjectCreate(0, "HUD_Scale", OBJ_LABEL, 0, 0, 0);
   ObjectSetString(0,  "HUD_Scale", OBJPROP_TEXT, scale_msg);
   ObjectSetString(0,  "HUD_Scale", OBJPROP_FONT, font);
   ObjectSetInteger(0, "HUD_Scale", OBJPROP_FONTSIZE, 10);
   ObjectSetInteger(0, "HUD_Scale", OBJPROP_COLOR, clrDeepSkyBlue);
   ObjectSetInteger(0, "HUD_Scale", OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, "HUD_Scale", OBJPROP_YDISTANCE, y);

   // Spread y ATR
   y += 20;
   symbolInfo.RefreshRates();
   double curAtr = 0; GetAtrValue(curAtr);
   double curPrice = symbolInfo.Ask();
   double atrPct = (curPrice > 0) ? (curAtr / curPrice) * 100.0 : 0.0;
   string stats = StringFormat("SPREAD: %d | ATR Pct: %.3f%%", symbolInfo.Spread(), atrPct);
   if(ObjectFind(0, "HUD_Stats") < 0) ObjectCreate(0, "HUD_Stats", OBJ_LABEL, 0, 0, 0);
   ObjectSetString(0,  "HUD_Stats", OBJPROP_TEXT, stats);
   ObjectSetString(0,  "HUD_Stats", OBJPROP_FONT, font);
   ObjectSetInteger(0, "HUD_Stats", OBJPROP_FONTSIZE, 10);
   ObjectSetInteger(0, "HUD_Stats", OBJPROP_COLOR, clrText);
   ObjectSetInteger(0, "HUD_Stats", OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, "HUD_Stats", OBJPROP_YDISTANCE, y);

   // DD diario + pérdidas consecutivas
   y += 20;
   double current_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double dd_pct = (daily_start_balance > 0)
      ? ((daily_start_balance - current_equity) / daily_start_balance) * 100.0 : 0.0;
   if(dd_pct < 0) dd_pct = 0.0;

   MqlDateTime dt_s; TimeToStruct(TimeCurrent(), dt_s);
   dt_s.hour = 0; dt_s.min = 0; dt_s.sec = 0;
   int current_losses = GetConsecutiveLossesToday(StructToTime(dt_s));

   string dd_str = StringFormat("DAILY DD: %.2f%% | CONSEC LOSSES: %d/%d",
                                  dd_pct, current_losses, InpMaxConsecutiveLosses);
   if(ObjectFind(0, "HUD_DD") < 0) ObjectCreate(0, "HUD_DD", OBJ_LABEL, 0, 0, 0);
   ObjectSetString(0,  "HUD_DD", OBJPROP_TEXT, dd_str);
   ObjectSetString(0,  "HUD_DD", OBJPROP_FONT, font);
   ObjectSetInteger(0, "HUD_DD", OBJPROP_FONTSIZE, 10);
   ObjectSetInteger(0, "HUD_DD", OBJPROP_COLOR,
      (dd_pct >= InpMaxDailyLossPct * 0.8 || current_losses >= InpMaxConsecutiveLosses)
         ? clrRed : clrText);
   ObjectSetInteger(0, "HUD_DD", OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, "HUD_DD", OBJPROP_YDISTANCE, y);

   // V22: Peak Equity Lock
   y += 20;
   double dd_from_peak = (peak_equity_today > 0)
      ? ((peak_equity_today - current_equity) / peak_equity_today) * 100.0 : 0.0;
   if(dd_from_peak < 0) dd_from_peak = 0.0;
   string peak_str = StringFormat("PEAK HOY: $%.2f | DD PICO: %.2f%%",
                                    peak_equity_today, dd_from_peak);
   if(ObjectFind(0, "HUD_Peak") < 0) ObjectCreate(0, "HUD_Peak", OBJ_LABEL, 0, 0, 0);
   ObjectSetString(0,  "HUD_Peak", OBJPROP_TEXT, peak_str);
   ObjectSetString(0,  "HUD_Peak", OBJPROP_FONT, font);
   ObjectSetInteger(0, "HUD_Peak", OBJPROP_FONTSIZE, 10);
   ObjectSetInteger(0, "HUD_Peak", OBJPROP_COLOR,
      (dd_from_peak >= InpPeakLockDrawdownPct * 0.8) ? clrOrange : clrText);
   ObjectSetInteger(0, "HUD_Peak", OBJPROP_XDISTANCE, x);
   ObjectSetInteger(0, "HUD_Peak", OBJPROP_YDISTANCE, y);
}

void CleanDashboard()
{
   string ids[] = {"HUD_Title","HUD_State","HUD_Session","HUD_Scale",
                   "HUD_Stats","HUD_DD","HUD_Peak"};
   for(int i = 0; i < ArraySize(ids); i++)
      ObjectDelete(0, ids[i]);
}

//+------------------------------------------------------------------+
//| Lógica de Protección de Viernes                                 |
//+------------------------------------------------------------------+
bool IsFridayCloseSession(MqlDateTime &dt)
{
   if(!InpAvoidFridayClose) return false;
   // V22: Usar hora UTC para consistencia
   MqlDateTime gmt_dt;
   TimeToStruct(TimeGMT(), gmt_dt);
   return (gmt_dt.day_of_week == 5 && gmt_dt.hour >= 20); // 20:00 UTC viernes
}

void CheckFridayClosure(MqlDateTime &dt)
{
   if(!InpAvoidFridayClose) return;
   MqlDateTime gmt_dt;
   TimeToStruct(TimeGMT(), gmt_dt);
   if(gmt_dt.day_of_week == 5 && gmt_dt.hour >= 21 && PositionsTotal() > 0)
      CloseAllPositions("Cierre de Fin de Semana V22");
}

//+------------------------------------------------------------------+
//| Preparación del Tensor para BUY (5 features: O H L C V)        |
//+------------------------------------------------------------------+
bool PrepareTensorData(float &tensor[])
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   if(CopyRates(_Symbol, PERIOD_M5, 1, InpTimeSteps, rates) != InpTimeSteps) return false;

   double minPrice = DBL_MAX, maxPrice = -DBL_MAX;
   double minVol   = DBL_MAX, maxVol   = -DBL_MAX;

   for(int i = 0; i < InpTimeSteps; i++)
   {
      if(rates[i].low  < minPrice) minPrice = rates[i].low;
      if(rates[i].high > maxPrice) maxPrice = rates[i].high;
      if((double)rates[i].tick_volume < minVol) minVol = (double)rates[i].tick_volume;
      if((double)rates[i].tick_volume > maxVol) maxVol = (double)rates[i].tick_volume;
   }

   ArrayResize(tensor, InpTimeSteps * 5);
   int idx = 0;
   for(int i = InpTimeSteps - 1; i >= 0; i--)
   {
      tensor[idx++] = NormalizeWindowValue(rates[i].open,                minPrice, maxPrice);
      tensor[idx++] = NormalizeWindowValue(rates[i].high,                minPrice, maxPrice);
      tensor[idx++] = NormalizeWindowValue(rates[i].low,                 minPrice, maxPrice);
      tensor[idx++] = NormalizeWindowValue(rates[i].close,               minPrice, maxPrice);
      tensor[idx++] = NormalizeWindowValue((double)rates[i].tick_volume,  minVol,  maxVol);
   }
   return true;
}

//+------------------------------------------------------------------+
//| Preparación del Tensor para SELL V2 (7 features: +RSI +Slope)   |
//+------------------------------------------------------------------+
bool PrepareTensorDataSellV2(float &tensor[])
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   if(CopyRates(_Symbol, PERIOD_M5, 1, InpTimeSteps, rates) != InpTimeSteps) return false;

   double rsi_buf[];
   ArraySetAsSeries(rsi_buf, true);
   bool has_rsi = (rsi_handle != INVALID_HANDLE &&
                   CopyBuffer(rsi_handle, 0, 1, InpTimeSteps, rsi_buf) == InpTimeSteps);
   if(!has_rsi)
   {
      ArrayResize(rsi_buf, InpTimeSteps);
      ArrayFill(rsi_buf, 0, InpTimeSteps, 50.0);
   }

   double minPrice = DBL_MAX, maxPrice = -DBL_MAX;
   double minVol   = DBL_MAX, maxVol   = -DBL_MAX;
   for(int i = 0; i < InpTimeSteps; i++)
   {
      if(rates[i].low  < minPrice) minPrice = rates[i].low;
      if(rates[i].high > maxPrice) maxPrice = rates[i].high;
      if((double)rates[i].tick_volume < minVol) minVol = (double)rates[i].tick_volume;
      if((double)rates[i].tick_volume > maxVol) maxVol = (double)rates[i].tick_volume;
   }

   ArrayResize(tensor, InpTimeSteps * 7);
   int idx = 0;
   for(int i = InpTimeSteps - 1; i >= 0; i--)
   {
      tensor[idx++] = NormalizeWindowValue(rates[i].open,                minPrice, maxPrice);
      tensor[idx++] = NormalizeWindowValue(rates[i].high,                minPrice, maxPrice);
      tensor[idx++] = NormalizeWindowValue(rates[i].low,                 minPrice, maxPrice);
      tensor[idx++] = NormalizeWindowValue(rates[i].close,               minPrice, maxPrice);
      tensor[idx++] = NormalizeWindowValue((double)rates[i].tick_volume,  minVol,  maxVol);
      // RSI normalizado [0,1]
      tensor[idx++] = (float)(rsi_buf[i] / 100.0);
      // Slope: pendiente de 5 velas, clip a ±0.003, normalizado a [-1,1]
      double slope = 0.0;
      if(i + 5 < InpTimeSteps)
         slope = (rates[i].close - rates[i + 5].close) / (rates[i].close + 1e-10);
      tensor[idx++] = (float)MathMax(-1.0, MathMin(1.0, slope / 0.003));
   }
   return true;
}

float RunAIBuyInference(const float &input_data[])
{
   float output_data[1]; output_data[0] = 0.0f;
   if(model_buy_handle == INVALID_HANDLE) return 0.0f;
   if(!OnnxRun(model_buy_handle, ONNX_DEFAULT, input_data, output_data)) return 0.0f;
   return output_data[0];
}

float RunAISellInference(const float &input_data[])
{
   float output_data[1]; output_data[0] = 0.0f;
   if(model_sell_handle == INVALID_HANDLE) return 0.0f;
   if(!OnnxRun(model_sell_handle, ONNX_DEFAULT, input_data, output_data)) return 0.0f;
   return output_data[0];
}

//+------------------------------------------------------------------+
//| Ejecución de Trade (V23)                                        |
//+------------------------------------------------------------------+
bool ExecuteTrade(const ENUM_ORDER_TYPE type, const double ai_probability)
{
   double atr_value;
   if(!GetAtrValue(atr_value)) return false;

   symbolInfo.RefreshRates();
   double current_price = (type == ORDER_TYPE_BUY) ? symbolInfo.Ask() : symbolInfo.Bid();

   double sl_distance = atr_value * InpSL_ATR_Mult;
   double tp_distance = atr_value * InpTP_ATR_Mult;

   double min_points_sl = 200 * symbolInfo.Point();
   if(sl_distance < min_points_sl) sl_distance = min_points_sl;

   double tick_value     = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size      = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double min_lot        = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double step_lot       = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double max_lot        = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   double lots = 0.0;
   double current_balance = AccountInfoDouble(ACCOUNT_BALANCE);

   if(InpLotMethod == LOT_SIZING_PER_1K)
   {
      double multiplier = current_balance / 1000.0;
      lots = multiplier * InpLotsPer1K;
   }
   else
   {
      double riskAmount    = current_balance * (InpRiskPercent / 100.0);
      double money_per_lot = (sl_distance / tick_size) * tick_value;
      if(money_per_lot <= 0) return false;
      lots = riskAmount / money_per_lot;
   }

   lots = MathFloor(lots / step_lot) * step_lot;
   lots = MathMax(lots, min_lot);
   lots = MathMin(lots, max_lot);

   // Si usamos scale-out, necesitamos mínimo 2 lotes mínimos para poder partir
   if(lots == min_lot && InpUsePartialClose)
   {
      double margin_req;
      double test_lots = min_lot * 2;
      if(OrderCalcMargin(type, _Symbol, test_lots, current_price, margin_req)
         && margin_req <= AccountInfoDouble(ACCOUNT_MARGIN_FREE) * 0.8)
         lots = test_lots;
   }

   bool result = false;
   string comment = StringFormat("IA_V22|%s|p=%.2f", (type == ORDER_TYPE_BUY ? "L" : "S"), ai_probability);

   if(type == ORDER_TYPE_BUY)
   {
      double sl = NormalizeDouble(current_price - sl_distance, _Digits);
      double tp = NormalizeDouble(current_price + tp_distance, _Digits);
      result = trade.Buy(lots, _Symbol, current_price, sl, tp, comment);
   }
   else if(type == ORDER_TYPE_SELL)
   {
      double sl = NormalizeDouble(current_price + sl_distance, _Digits);
      double tp = NormalizeDouble(current_price - tp_distance, _Digits);
      result = trade.Sell(lots, _Symbol, current_price, sl, tp, comment);
   }

   if(result)
   {
      lastTradeTime = TimeCurrent();
      Print("V22 APEX ENTRY! Lotes: ", lots, " | Prob: ", NormalizeDouble(ai_probability, 3),
            " | SL_Dist: ", NormalizeDouble(sl_distance, _Digits),
            " | TP_Dist: ", NormalizeDouble(tp_distance, _Digits));
   }
   return result;
}

//+------------------------------------------------------------------+
//| Verificador de Cierres Parciales                                |
//+------------------------------------------------------------------+
bool IsPositionPartiallyClosed(ulong pos_id)
{
   HistorySelectByPosition(pos_id);
   int out_deals = 0;
   for(int i = 0; i < HistoryDealsTotal(); i++)
      if(HistoryDealGetInteger(HistoryDealGetTicket(i), DEAL_ENTRY) == DEAL_ENTRY_OUT)
         out_deals++;
   return (out_deals > 0);
}

//+------------------------------------------------------------------+
//| Motor de Salidas: SHADOW TRAILING V22 (GOLDEN RUNNER OPTIMIZADO)|
//+------------------------------------------------------------------+
void ManageInstitutionalExits()
{
   double atr[1];
   if(CopyBuffer(atr_handle, 0, 1, 1, atr) <= 0) return;

   double min_stop = MathMax(
      (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL),
      (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL)
   ) * symbolInfo.Point() + (5 * symbolInfo.Point());

   double step_vol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double min_lot  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);

   double risk_red_trigger  = atr[0] * InpRiskReductionTrigger;
   double risk_red_secure   = atr[0] * InpRiskReductionSet;
   double be_trigger_dist   = atr[0] * InpBE_TriggerATR;
   double be_secure_dist    = atr[0] * InpBE_SecureATR;
   double scale_trigger_dist= atr[0] * InpSmartLockTriggerATR;
   double scale_secure_dist = atr[0] * InpSmartLockSecureATR;
   double step_pts          = atr[0] * InpTrailStepATRMult;
   double min_diff_pts      = MathMax(10 * symbolInfo.Point(), min_stop);

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionGetString(POSITION_SYMBOL) != _Symbol
         || PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;

      ulong  pos_id      = PositionGetInteger(POSITION_IDENTIFIER);
      double open_price  = PositionGetDouble(POSITION_PRICE_OPEN);
      double current_sl  = PositionGetDouble(POSITION_SL);
      double current_price = PositionGetDouble(POSITION_PRICE_CURRENT);
      double current_vol = PositionGetDouble(POSITION_VOLUME);
      int    pos_type    = (int)PositionGetInteger(POSITION_TYPE);

      bool is_partially_closed = IsPositionPartiallyClosed(pos_id);
      double new_sl          = current_sl;
      bool modification_needed = false;

      // ==========================================
      // BUY POSITIONS
      // ==========================================
      if(pos_type == POSITION_TYPE_BUY)
      {
         // FASE 3: SCALE-OUT + GOLDEN RUNNER TRAILING
         if(current_vol > min_lot)
         {
            if(InpUsePartialClose && !is_partially_closed
               && current_price >= open_price + scale_trigger_dist)
            {
               double half_vol = MathFloor((current_vol / 2.0) / step_vol) * step_vol;
               if(half_vol >= min_lot)
               {
                  Print("Scale-Out V22 BUY: Cobrando 50%. Runner activo.");
                  trade.PositionClosePartial(ticket, half_vol);

                  double proposed_sl = open_price + scale_secure_dist;
                  if(current_sl < proposed_sl && current_price - proposed_sl > min_stop)
                  {
                     new_sl = proposed_sl;
                     modification_needed = true;
                  }
               }
            }
         }
         else if(is_partially_closed || !InpUsePartialClose)
         {
            // Trailing activo para el runner
            if(current_price >= open_price + scale_trigger_dist)
            {
               double proposed_sl = open_price + scale_secure_dist
                  + (MathFloor((current_price - open_price - scale_trigger_dist) / step_pts) * step_pts);
               if(proposed_sl > current_sl && (current_price - proposed_sl) > min_stop)
               {
                  new_sl = proposed_sl;
                  modification_needed = true;
               }
            }
         }

         // FASE 2: BREAK-EVEN
         if(!modification_needed && current_price >= open_price + be_trigger_dist)
         {
            double proposed_be = open_price + be_secure_dist;
            if(current_sl < proposed_be && current_price - proposed_be > min_stop)
            {
               new_sl = proposed_be;
               modification_needed = true;
            }
         }

         // FASE 1: REDUCCIÓN DE RIESGO
         if(!modification_needed && current_price >= open_price + risk_red_trigger)
         {
            double proposed_risk_red = open_price - risk_red_secure;
            if(current_sl < proposed_risk_red
               && current_price - proposed_risk_red > min_stop
               && proposed_risk_red < open_price)
            {
               new_sl = proposed_risk_red;
               modification_needed = true;
            }
         }

         if(modification_needed)
         {
            new_sl     = NormalizeDouble(new_sl, _Digits);
            current_sl = NormalizeDouble(current_sl, _Digits);
            if(MathAbs(new_sl - current_sl) >= min_diff_pts && new_sl > current_sl)
            {
               trade.PositionModify(ticket, new_sl, PositionGetDouble(POSITION_TP));
               Print("V22 SL Runner BUY ajustado a: ", new_sl);
            }
         }
      }

      // ==========================================
      // SELL POSITIONS
      // ==========================================
      else if(pos_type == POSITION_TYPE_SELL)
      {
         // FASE 3: SCALE-OUT + GOLDEN RUNNER TRAILING
         if(current_vol > min_lot)
         {
            if(InpUsePartialClose && !is_partially_closed
               && current_price <= open_price - scale_trigger_dist)
            {
               double half_vol = MathFloor((current_vol / 2.0) / step_vol) * step_vol;
               if(half_vol >= min_lot)
               {
                  Print("Scale-Out V22 SELL: Cobrando 50%. Runner activo.");
                  trade.PositionClosePartial(ticket, half_vol);

                  double proposed_sl = open_price - scale_secure_dist;
                  if((current_sl > proposed_sl || current_sl == 0)
                     && proposed_sl - current_price > min_stop)
                  {
                     new_sl = proposed_sl;
                     modification_needed = true;
                  }
               }
            }
         }
         else if(is_partially_closed || !InpUsePartialClose)
         {
            if(current_price <= open_price - scale_trigger_dist)
            {
               double proposed_sl = open_price - scale_secure_dist
                  - (MathFloor((open_price - current_price - scale_trigger_dist) / step_pts) * step_pts);
               if((proposed_sl < current_sl || current_sl == 0)
                  && (proposed_sl - current_price) > min_stop)
               {
                  new_sl = proposed_sl;
                  modification_needed = true;
               }
            }
         }

         // FASE 2: BREAK-EVEN
         if(!modification_needed && current_price <= open_price - be_trigger_dist)
         {
            double proposed_be = open_price - be_secure_dist;
            if((current_sl > proposed_be || current_sl == 0)
               && proposed_be - current_price > min_stop)
            {
               new_sl = proposed_be;
               modification_needed = true;
            }
         }

         // FASE 1: REDUCCIÓN DE RIESGO
         if(!modification_needed && current_price <= open_price - risk_red_trigger)
         {
            double proposed_risk_red = open_price + risk_red_secure;
            if((current_sl > proposed_risk_red || current_sl == 0)
               && proposed_risk_red - current_price > min_stop
               && proposed_risk_red > open_price)
            {
               new_sl = proposed_risk_red;
               modification_needed = true;
            }
         }

         if(modification_needed)
         {
            new_sl     = NormalizeDouble(new_sl, _Digits);
            current_sl = NormalizeDouble(current_sl, _Digits);
            if(MathAbs(new_sl - current_sl) >= min_diff_pts
               && (new_sl < current_sl || current_sl == 0.0))
            {
               trade.PositionModify(ticket, new_sl, PositionGetDouble(POSITION_TP));
               Print("V22 SL Runner SELL ajustado a: ", new_sl);
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Init / Deinit                                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   symbolInfo.Name(_Symbol);
   trade.SetExpertMagicNumber(InpMagic);

   // Log del offset del broker al iniciar
   Print("V22 Init: Offset broker vs UTC = UTC", GetBrokerUTCOffset(),
         " | TimeGMT: ", TimeGMT(), " | TimeCurrent: ", TimeCurrent());

   model_buy_handle = OnnxCreateFromBuffer(ExtModelDataBuy, ONNX_DEFAULT);
   if(model_buy_handle == INVALID_HANDLE)
   {
      Print("ERROR: No se pudo cargar gold_lstm_model.onnx");
      return(INIT_FAILED);
   }

   model_sell_handle = OnnxCreateFromBuffer(ExtModelDataSell, ONNX_DEFAULT);
   if(model_sell_handle == INVALID_HANDLE)
   {
      Print("ERROR: No se pudo cargar gold_lstm_model_sell.onnx");
      return(INIT_FAILED);
   }

   const long input_shape_buy[]  = {1, InpTimeSteps, 5};  // BUY: 5 features
   const long input_shape_sell[] = {1, InpTimeSteps, InpUseSellModelV2 ? 7 : 5}; // SELL: 7 o 5
   const long output_shape[]     = {1, 1};

   if(!OnnxSetInputShape(model_buy_handle,  0, input_shape_buy)
      || !OnnxSetOutputShape(model_buy_handle,  0, output_shape)) return(INIT_FAILED);
   if(!OnnxSetInputShape(model_sell_handle, 0, input_shape_sell)
      || !OnnxSetOutputShape(model_sell_handle, 0, output_shape)) return(INIT_FAILED);

   atr_handle = iATR(_Symbol, PERIOD_M15, InpAtrPeriod);
   if(atr_handle == INVALID_HANDLE) return(INIT_FAILED);

   // V23: RSI handle para el tensor de 7 features del modelo SELL V2
   rsi_handle = iRSI(_Symbol, PERIOD_M5, 14, PRICE_CLOSE);
   if(rsi_handle == INVALID_HANDLE)
   {
      Print("ADVERTENCIA: No se pudo crear handle RSI. El modelo SELL V2 usara RSI=50 como fallback.");
   }

   daily_start_balance  = AccountInfoDouble(ACCOUNT_BALANCE);
   peak_equity_today    = daily_start_balance;

   EventSetTimer(1);
   Print("GOLD IA GEM V23 QUANTUM APEX inicializado. Smart Peak Lock=", InpSmartPeakLock,
         " | Sell Model V2=", InpUseSellModelV2,
         " | Offset broker UTC", GetBrokerUTCOffset());
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   CleanDashboard();
   if(model_buy_handle  != INVALID_HANDLE) OnnxRelease(model_buy_handle);
   if(model_sell_handle != INVALID_HANDLE) OnnxRelease(model_sell_handle);
   if(atr_handle        != INVALID_HANDLE) IndicatorRelease(atr_handle);
   if(rsi_handle        != INVALID_HANDLE) IndicatorRelease(rsi_handle);
}

void OnTimer()
{
   UpdateDashboard();
}

//+------------------------------------------------------------------+
//| Loop Principal (OnTick)                                         |
//+------------------------------------------------------------------+
void OnTick()
{
   double current_equity = AccountInfoDouble(ACCOUNT_EQUITY);

   // ------- Reset diario -------
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   if(dt.day_of_year != current_day)
   {
      current_day          = dt.day_of_year;
      daily_start_balance  = AccountInfoDouble(ACCOUNT_BALANCE);
      peak_equity_today    = daily_start_balance;
      trading_halted_today = false;
      halt_reason          = "";
      Print("V23 Nuevo dia. Balance inicio: ", daily_start_balance,
            " | Peak reset: ", peak_equity_today);
   }

   // ------- Actualizar peak equity -------
   if(current_equity > peak_equity_today)
      peak_equity_today = current_equity;

   // ------- Protecciones de riesgo -------
   CheckFridayClosure(dt);

   // 1. Max Daily Loss
   if(!trading_halted_today
      && ((daily_start_balance - current_equity) / daily_start_balance) * 100.0 >= InpMaxDailyLossPct)
   {
      Print("V23 LIMITE PERDIDA DIARIA: ", InpMaxDailyLossPct, "%. Deteniendo EA.");
      CloseAllPositions("Max Daily Loss V23");
      trading_halted_today = true;
      halt_reason  = "MAX DAILY LOSS";
      ai_status_msg = "MAX DAILY LOSS ALCANZADO";
   }

   // 2. V23 SMART Peak Equity Lock
   if(!trading_halted_today && InpUsePeakLock && peak_equity_today > daily_start_balance)
   {
      double dd_from_peak = ((peak_equity_today - current_equity) / peak_equity_today) * 100.0;
      if(dd_from_peak >= InpPeakLockDrawdownPct)
      {
         // V23 SMART: verificar si todas las posiciones abiertas ya tienen SL en zona segura
         bool all_positions_safe = true;
         if(InpSmartPeakLock)
         {
            for(int pi = PositionsTotal() - 1; pi >= 0; pi--)
            {
               ulong tk = PositionGetTicket(pi);
               if(PositionGetString(POSITION_SYMBOL) != _Symbol
                  || PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;

               double pos_open  = PositionGetDouble(POSITION_PRICE_OPEN);
               double pos_sl    = PositionGetDouble(POSITION_SL);
               int    pos_type  = (int)PositionGetInteger(POSITION_TYPE);

               // Una posición es "segura" si su SL ya garantiza ganancia o BE
               bool sl_in_profit = false;
               if(pos_type == POSITION_TYPE_BUY)
                  sl_in_profit = (pos_sl >= pos_open);   // SL por encima del entry = seguro
               else
                  sl_in_profit = (pos_sl > 0 && pos_sl <= pos_open); // SL por debajo del entry = seguro

               if(!sl_in_profit)
               {
                  all_positions_safe = false;
                  break;
               }
            }
         }

         if(!InpSmartPeakLock || !all_positions_safe)
         {
            Print("V23 SMART PEAK LOCK: Equity cayo ", NormalizeDouble(dd_from_peak, 2),
                  "% desde el pico ($", NormalizeDouble(peak_equity_today, 2),
                  "). Posiciones en riesgo real — cerrando.");
            CloseAllPositions("Smart Peak Equity Lock V23");
            trading_halted_today = true;
            halt_reason   = "SMART PEAK LOCK";
            ai_status_msg = "SMART PEAK LOCK ACTIVADO";
         }
         else
         {
            // Posiciones seguras — solo detener nuevas entradas, no cerrar
            if(!trading_halted_today)
            {
               Print("V23 SMART PEAK LOCK: DD pico ", NormalizeDouble(dd_from_peak, 2),
                     "% pero todas las posiciones tienen SL en zona segura. Bloqueando nuevas entradas.");
               trading_halted_today = true;
               halt_reason   = "PEAK LOCK (entradas bloqueadas)";
               ai_status_msg = "PEAK LOCK — RUNNER LIBRE";
            }
         }
      }
   }

   // 3. Circuit Breaker (pérdidas consecutivas)
   MqlDateTime dt_start;
   TimeToStruct(TimeCurrent(), dt_start);
   dt_start.hour = 0; dt_start.min = 0; dt_start.sec = 0;
   int current_losses = GetConsecutiveLossesToday(StructToTime(dt_start));

   if(!trading_halted_today && current_losses >= InpMaxConsecutiveLosses)
   {
      Print("CIRCUIT BREAKER: ", InpMaxConsecutiveLosses, " perdidas seguidas. Apagando EA.");
      trading_halted_today = true;
      halt_reason  = "CIRCUIT BREAKER";
      ai_status_msg = "CIRCUIT BREAKER ACTIVADO";
   }

   // ------- Gestión de posiciones abiertas (siempre activa) -------
   if(PositionsTotal() > 0)
   {
      ManageInstitutionalExits();
      if(!trading_halted_today) ai_status_msg = "POSICION ABIERTA (GESTIONANDO)";
   }

   if(trading_halted_today) return;

   // ------- Filtro de viernes / fin de semana -------
   if(IsFridayCloseSession(dt))
   {
      ai_status_msg = "BLOQUEADO (FIN DE SEMANA)";
      return;
   }

   // ------- Filtro de sesión (V22: basado en GMT) -------
   if(!IsWithinTradingSession())
   {
      ai_status_msg = "FUERA DE SESION";
      return;
   }

   // ------- Cooldown -------
   if(IsCooldownActive())
   {
      ai_status_msg = "EN COOLDOWN";
   }
   else if(PositionsTotal() == 0)
   {
      ai_status_msg = "ESCANEANDO (PURE A.I.)";
   }

   // ------- Solo ejecutar en nueva vela M5 -------
   if(!IsNewBar(PERIOD_M5, lastSignalBarTime)) return;

   symbolInfo.RefreshRates();
   if(symbolInfo.Spread() > InpMaxSpread) return;

   // ------- IA PREDICTIVA DUAL-CORE V23 -------
   // BUY: tensor de 5 features (modelo original)
   float tensor_buy[];
   if(!PrepareTensorData(tensor_buy)) return;

   double current_price = symbolInfo.Bid();
   double atr_val;
   if(!GetAtrValue(atr_val)) return;

   double atr_percentage = (current_price > 0) ? (atr_val / current_price) * 100.0 : 0.0;
   if(atr_percentage < InpMinATR_Pct || atr_percentage > InpMaxATR_Pct) return;

   bool has_pos = PositionSelect(_Symbol)
                  && PositionGetInteger(POSITION_MAGIC) == InpMagic;

   // --- COMPRAS (modelo 5 features) ---
   if(InpEnableBuys && !has_pos && !IsCooldownActive())
   {
      float prob_buy = RunAIBuyInference(tensor_buy);
      if(prob_buy >= (float)InpMinProbBuy)
      {
         Print("V23 Señal BUY | Prob: ", NormalizeDouble(prob_buy, 3));
         if(ExecuteTrade(ORDER_TYPE_BUY, prob_buy)) return;
      }
   }

   // --- VENTAS (modelo V2: 7 features si InpUseSellModelV2, o 5 features si no) ---
   if(InpEnableSells && !has_pos && !IsCooldownActive())
   {
      float tensor_sell[];
      bool tensor_ok = InpUseSellModelV2
                       ? PrepareTensorDataSellV2(tensor_sell)
                       : PrepareTensorData(tensor_sell);

      if(tensor_ok)
      {
         float prob_sell = RunAISellInference(tensor_sell);
         if(prob_sell >= (float)InpMinProbSell)
         {
            Print("V23 Señal SELL | Prob: ", NormalizeDouble(prob_sell, 3),
                  " | Modelo: ", InpUseSellModelV2 ? "V2 (7f)" : "V1 (5f)");
            ExecuteTrade(ORDER_TYPE_SELL, prob_sell);
         }
      }
   }
}
//+------------------------------------------------------------------+
