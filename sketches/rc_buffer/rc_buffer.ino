// Task 4: RC step response with BUFFERED, fixed-period sampling.
//
// Wiring: D7 -> R -> node -> C -> GND ; node -> A0.
// The program switches D7 itself (no manual switch) at sample i = PRE, stores N
// samples in RAM at a fixed period DT_US (timed with micros()) and only then sends
// them over serial. The serial line therefore no longer sets the sampling rate.
//
// Configuration is sent from the PC after reset (no recompiling per resistor):
//   "c <dt_us> <settle_ms> <reps>\n"
// Output: "# key=value" metadata, "# clock us=<micros>" lines (the PC stamps them
// with its own time to calibrate the resonator), "phase,rep,i,adc" rows, "END".
// phase R = D7 LOW->HIGH (charging), F = HIGH->LOW (discharging).

const uint8_t DRIVE = 7;
const uint8_t SENSE = A0;
const uint16_t N = 500;      // samples per capture (2 bytes each in RAM)
const uint16_t PRE = 50;     // samples before the step (baseline / noise)

uint16_t buf[N];
unsigned long DT_US = 500, SETTLE_MS = 2000;
uint8_t REPS = 3;
unsigned int overruns = 0;
unsigned long max_late = 0;

void clockStamp() {
  Serial.print("# clock us=");
  Serial.println(micros());
}

void capture(uint8_t level) {
  overruns = 0;
  max_late = 0;
  unsigned long t0 = micros();
  for (uint16_t i = 0; i < N; i++) {
    unsigned long target = (unsigned long)i * DT_US;
    while (micros() - t0 < target) {}
    unsigned long late = micros() - t0 - target;   // how late this sample started
    if (late > max_late) max_late = late;
    if (late > DT_US / 4) overruns++;
    if (i == PRE) digitalWrite(DRIVE, level);
    buf[i] = analogRead(SENSE);
  }
}

void dump(char phase, uint8_t rep) {
  Serial.print("# capture ");
  Serial.print(phase);
  Serial.print(rep);
  Serial.print(" overruns=");
  Serial.print(overruns);
  Serial.print(" max_late_us=");
  Serial.println(max_late);
  for (uint16_t i = 0; i < N; i++) {
    Serial.print(phase); Serial.print(',');
    Serial.print(rep);   Serial.print(',');
    Serial.print(i);     Serial.print(',');
    Serial.println(buf[i]);
  }
}

void setup() {
  pinMode(DRIVE, OUTPUT);
  digitalWrite(DRIVE, LOW);
  Serial.begin(115200);
  Serial.println("# ready");

  // wait for "c <dt_us> <settle_ms> <reps>"
  while (true) {
    if (Serial.available() && Serial.read() == 'c') {
      DT_US = Serial.parseInt();
      SETTLE_MS = Serial.parseInt();
      REPS = Serial.parseInt();
      break;
    }
  }

  analogRead(SENSE);   // discard the first conversion
  unsigned long t = micros();
  for (int k = 0; k < 1000; k++) analogRead(SENSE);
  float adc_us = (micros() - t) / 1000.0;

  Serial.print("# dt_us=");      Serial.print(DT_US);
  Serial.print(" n=");           Serial.print(N);
  Serial.print(" pre=");         Serial.print(PRE);
  Serial.print(" reps=");        Serial.print(REPS);
  Serial.print(" settle_ms=");   Serial.print(SETTLE_MS);
  Serial.print(" drive_pin=");   Serial.print(DRIVE);
  Serial.print(" adc_read_us="); Serial.println(adc_us, 1);
  clockStamp();

  for (uint8_t rep = 0; rep < REPS; rep++) {
    digitalWrite(DRIVE, LOW);        // start discharged
    delay(SETTLE_MS);
    capture(HIGH);
    dump('R', rep);
    clockStamp();
    delay(SETTLE_MS);
    capture(LOW);
    dump('F', rep);
    clockStamp();
  }
  digitalWrite(DRIVE, LOW);
  Serial.println("END");
}

void loop() {}
