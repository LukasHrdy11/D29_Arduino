// Task 3: RC charging/discharging, ORIGINAL method of the assignment (obr. 16,
// ReadAnalogVoltageRC): 9600 baud, no delay, each line "millis;voltage".
// Kept deliberately unchanged so that task 4 (buffered sampling) can be compared
// with it: the sampling period is set by the serial line, not by the program.

void setup() {
  Serial.begin(9600);
}

void loop() {
  int sensorValue = analogRead(A0);
  float voltage = sensorValue * (5.0 / 1023.0);
  Serial.print(millis());
  Serial.print(";");
  Serial.println(voltage);          // default: 2 decimal places
}
