// Test simple PING-PONG + relay (simple test) #yesssssss
const int relays[] = {7,8,9,10};
const bool ACTIVE_LOW = false;
void setup() {
  Serial.begin(115200);
  for (int i=0;i<4;i++){ pinMode(relays[i], OUTPUT); digitalWrite(relays[i], ACTIVE_LOW?HIGH:LOW); }
  Serial.println("Relay Controller Ready");
}
String s="";
void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c=='\n') {
      s.trim();
      if (s=="PING") Serial.println("PONG");
      if (s.startsWith("SET")) {
        // parse SET n v
        int a = s.indexOf(' ',4);
        int n = s.substring(4,a).toInt();
        int v = s.substring(a+1).toInt();
        if (n>=1 && n<=4) {
          if (ACTIVE_LOW) digitalWrite(relays[n-1], v?LOW:HIGH);
          else digitalWrite(relays[n-1], v?HIGH:LOW);
          Serial.print("OK ");
          Serial.print(n);
          Serial.print(" ");
          Serial.println(v);
        }
      }
      s="";
    } else if (c!='\r') {
      s += c;
    }
  }
}
