// D29 feasibility test: continuous fixed-rate streaming with a modulated LED.
//
// Wiring (same as TEST 2):
//   D9 -> 220R -> LED(+, long leg) ; LED(-) -> GND
//   5V -> LDR -> node -> 10k -> GND ; node -> A0
//
// Every sample period the sketch outputs one line "ref,adc":
//   ref = current LED state (0/1), i.e. the lock-in reference,
//   adc = analogRead(A0), 0..1023.
// The PC (capture.py) does all the lock-in maths, so nothing has to be
// decided on the Arduino yet.
//
// Commands over serial (newline terminated):
//   f<Hz>   set modulation frequency, e.g. "f7.3";  "f0" = LED off (noise floor)
//   h       LED constantly on (DC level)
//   p       print status line "# fs=... f=... overruns=..."
// Status lines start with '#', so the parser can skip them.

const uint8_t LED_PIN = 9;
const uint8_t SENSE = A0;
const unsigned long DT_US = 1000;   // 1 kHz sampling; "1,1023\n" = 7 B -> 7 kB/s < 11.5 kB/s

float f_mod = 0.0;                  // modulation frequency [Hz], 0 = off
bool constant_on = false;
unsigned long half_period_samples = 0;
unsigned long sample_idx = 0;
unsigned long overruns = 0;
uint8_t ref = 0;
char cmd[16];
uint8_t cmd_len = 0;

void apply_freq(float f) {
  f_mod = f;
  constant_on = false;
  if (f > 0) {
    // the LED toggles on a whole number of samples -> the actual frequency
    // is quantized; report it so the PC uses the true value
    half_period_samples = (unsigned long)(1e6 / (2.0 * f * DT_US) + 0.5);
    if (half_period_samples < 1) half_period_samples = 1;
  } else {
    half_period_samples = 0;
  }
  sample_idx = 0;
  overruns = 0;
}

void print_status() {
  Serial.print("# fs=");
  Serial.print(1e6 / DT_US, 1);
  Serial.print(" f_req=");
  Serial.print(f_mod, 3);
  Serial.print(" f_true=");
  if (half_period_samples > 0)
    Serial.print(1e6 / (2.0 * half_period_samples * DT_US), 4);
  else
    Serial.print(0);
  Serial.print(" const_on=");
  Serial.print(constant_on);
  Serial.print(" overruns=");
  Serial.println(overruns);
}

void handle_cmd() {
  cmd[cmd_len] = 0;
  if (cmd[0] == 'f') { apply_freq(atof(cmd + 1)); print_status(); }
  else if (cmd[0] == 'h') { apply_freq(0); constant_on = true; print_status(); }
  else if (cmd[0] == 'p') { print_status(); }
  cmd_len = 0;
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  Serial.begin(115200);
  print_status();
}

void loop() {
  static unsigned long t_next = micros();

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') { if (cmd_len) handle_cmd(); }
    else if (cmd_len < sizeof(cmd) - 1) cmd[cmd_len++] = c;
  }

  while ((long)(micros() - t_next) < 0) {}
  if ((long)(micros() - t_next) > (long)(DT_US / 4)) overruns++;
  t_next += DT_US;

  if (constant_on) ref = 1;
  else if (half_period_samples == 0) ref = 0;
  else ref = ((sample_idx / half_period_samples) & 1) ? 0 : 1;
  digitalWrite(LED_PIN, ref);
  sample_idx++;

  int adc = analogRead(SENSE);
  Serial.print(ref);
  Serial.print(',');
  Serial.println(adc);
}
