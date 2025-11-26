// Relay controller for Arduino Nano
// Protocol: send lines like "SET 1 1\n" to turn ON relay 1 (1..4)
//           "SET 2 0\n" to turn OFF relay 2
// Also optional "ALL <bits>\n" where bits is string of 4 chars '0'/'1' e.g. "ALL 1010"

const int relays[] = {7, 8, 9, 10}; // IN1..IN4
const bool ACTIVE_LOW = true;      // set true if your module is active-low
const int N = 4;

String inputLine = "";

void setup() {
  Serial.begin(115200);
  for (int i = 0; i < N; ++i) {
    pinMode(relays[i], OUTPUT);
    // default off
    if (ACTIVE_LOW) digitalWrite(relays[i], HIGH);
    else digitalWrite(relays[i], LOW);
  }
  Serial.println("Relay Controller Ready");
}

void setRelay(int idx, int value) {
  if (idx < 1 || idx > N) return;
  int pin = relays[idx - 1];
  bool out;
  if (ACTIVE_LOW) out = (value == 0) ? HIGH : LOW;
  else out = (value == 1) ? HIGH : LOW;
  digitalWrite(pin, out);
  Serial.print("OK ");
  Serial.print(idx);
  Serial.print(" ");
  Serial.println(value);
}

void setAll(const char *bits) {
  for (int i = 0; i < N; ++i) {
    if (bits[i] == '1') setRelay(i+1, 1);
    else setRelay(i+1, 0);
  }
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      inputLine.trim();
      if (inputLine.length() > 0) {
        // parse
        if (inputLine.startsWith("SET ")) {
          // SET <n> <0/1>
          int sp1 = inputLine.indexOf(' ', 4);
          if (sp1 > 3) {
            String sIdx = inputLine.substring(4, sp1);
            String sVal = inputLine.substring(sp1+1);
            int idx = sIdx.toInt();
            int val = sVal.toInt();
            setRelay(idx, val);
          }
        } else if (inputLine.startsWith("ALL ")) {
          String bits = inputLine.substring(4);
          if (bits.length() >= N) {
            char buf[5];
            bits.substring(0, N).toCharArray(buf, N+1);
            setAll(buf);
          }
        } else if (inputLine == "PING") {
          Serial.println("PONG");
        }
      }
      inputLine = "";
    } else {
      inputLine += c;
      // cap long lines
      if (inputLine.length() > 100) inputLine = inputLine.substring(inputLine.length()-100);
    }
  }
}

