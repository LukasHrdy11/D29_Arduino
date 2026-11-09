// Side experiment to task 4: output resistance of the Uno pins D7 and D8.
//
// Wiring: D7 -> point B (A1 also there) -> R_L -> point F (A2 and D8 there).
// States (all four are measured, then all pins are released for cooling):
//   O_H  D8 = INPUT (no current), D7 HIGH : A1 = unloaded HIGH level (= Vcc)
//   O_L  D8 = INPUT,              D7 LOW  : A1 = unloaded LOW level  (= 0 V)
//   HL   D7 HIGH, D8 LOW  : current D7 -> R_L -> D8
//   LH   D7 LOW,  D8 HIGH : current D8 -> R_L -> D7
// Each line: "state,cycle,mean_A1,mean_A2" (mean of NAVG ADC readings, in codes).
// The current is only on for ~0.1 s per state, then everything is off for 0.3 s.

const uint8_t P7 = 7, P8 = 8;
const uint16_t NAVG = 256;
const uint8_t CYCLES = 30;

float meanRead(uint8_t ch) {
  analogRead(ch);                 // discard the first reading after the mux change
  delayMicroseconds(200);
  analogRead(ch);
  uint32_t s = 0;
  for (uint16_t i = 0; i < NAVG; i++) s += analogRead(ch);
  return s / (float)NAVG;
}

void release() {
  pinMode(P7, OUTPUT); digitalWrite(P7, LOW);
  pinMode(P8, INPUT);             // no current through R_L
}

void measure(const char *name, uint8_t cyc, int s7, int s8) {
  pinMode(P7, OUTPUT); digitalWrite(P7, s7);
  if (s8 < 0) pinMode(P8, INPUT);
  else { pinMode(P8, OUTPUT); digitalWrite(P8, s8); }
  delay(5);
  float a1 = meanRead(A1);
  float a2 = meanRead(A2);
  release();
  Serial.print(name); Serial.print(',');
  Serial.print(cyc);  Serial.print(',');
  Serial.print(a1, 3); Serial.print(',');
  Serial.println(a2, 3);
}

void setup() {
  release();
  Serial.begin(115200);
  Serial.print("# rpin navg="); Serial.print(NAVG);
  Serial.print(" cycles="); Serial.println(CYCLES);
  delay(500);
  for (uint8_t c = 0; c < CYCLES; c++) {
    measure("O_H", c, HIGH, -1);
    measure("O_L", c, LOW, -1);
    measure("HL", c, HIGH, LOW);
    delay(300);
    measure("LH", c, LOW, HIGH);
    delay(300);
  }
  release();
  Serial.println("END");
}

void loop() {}
