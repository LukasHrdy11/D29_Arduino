// Task 2: LED I-V characteristic. A0 = pot wiper (LED anode), A1 = LED cathode
// = top of the 220 ohm shunt. Prints block averages of NAVG sample pairs.
// Vcc is estimated once at startup from the internal 1.1 V bandgap.

const int NAVG = 64;

long readVccMilli() {
  ADMUX = _BV(REFS0) | _BV(MUX3) | _BV(MUX2) | _BV(MUX1);   // bandgap vs AVcc
  delay(5);
  long sum = 0;
  for (int i = 0; i < 64; i++) {
    ADCSRA |= _BV(ADSC);
    while (ADCSRA & _BV(ADSC));
    if (i >= 8) sum += ADC;
  }
  return (long)(1100.0 * 1024.0 / (sum / 56.0));
}

void setup() {
  Serial.begin(115200);
  long vcc = readVccMilli();
  analogRead(A0);                     // restore mux, discard
  Serial.print("# vcc_mV=");
  Serial.println(vcc);
  Serial.print("# navg=");
  Serial.println(NAVG);
  Serial.println("# columns=t_ms,a0,a1");
}

void loop() {
  long s0 = 0, s1 = 0;
  for (int i = 0; i < NAVG; i++) {
    s0 += analogRead(A0);
    s1 += analogRead(A1);
  }
  Serial.print(millis());
  Serial.print(',');
  Serial.print(s0 / (float)NAVG, 2);
  Serial.print(',');
  Serial.println(s1 / (float)NAVG, 2);
}
