// Task 1.1: Blink with modified delays (on-board LED "L" = pin 13).
const int ON_MS  = 200;   // LED on
const int OFF_MS = 800;   // LED off

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  digitalWrite(LED_BUILTIN, HIGH);
  delay(ON_MS);
  digitalWrite(LED_BUILTIN, LOW);
  delay(OFF_MS);
}
