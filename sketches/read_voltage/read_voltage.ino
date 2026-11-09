// Task 1.2: ReadAnalogVoltage variant. Prints raw ADC code and voltage on A0.
// At startup measures Vcc from the internal 1.1 V bandgap (ADC reference = Vcc).

long readVccMilli() {
  // MUX = 1110 (bandgap 1.1 V), REFS0 = AVcc reference
  ADMUX = _BV(REFS0) | _BV(MUX3) | _BV(MUX2) | _BV(MUX1);
  delay(5);                                  // let the bandgap settle
  long sum = 0;
  for (int i = 0; i < 64; i++) {
    ADCSRA |= _BV(ADSC);
    while (ADCSRA & _BV(ADSC));
    if (i >= 8) sum += ADC;                  // discard first conversions
  }
  float code = sum / 56.0;
  return (long)(1100.0 * 1024.0 / code);     // nominal bandgap 1.1 V (+-10 %)
}

void setup() {
  Serial.begin(115200);
  long vcc = readVccMilli();
  analogRead(A0);                            // switch mux back, discard
  Serial.print("# vcc_mV=");
  Serial.println(vcc);
  Serial.println("# columns=adc,U_5V");
}

void loop() {
  int adc = analogRead(A0);
  float u = adc * (5.0 / 1023.0);            // as in the official example
  Serial.print(adc);
  Serial.print(',');
  Serial.println(u, 4);
  delay(20);
}
