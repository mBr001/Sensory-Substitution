int data;


void setup() { 
  Serial.begin(9600); 
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(12, OUTPUT);
  pinMode(11, OUTPUT);
  pinMode(10, OUTPUT);
  digitalWrite (LED_BUILTIN, LOW); //initially set to low
  Serial.println("Setting Up.");
}
 
void loop() {
while (Serial.available())
  {
    data = Serial.read();
  }

  if (data == '1') {
    digitalWrite (LED_BUILTIN, HIGH);
    digitalWrite (12, LOW);
    digitalWrite (11, LOW);
    digitalWrite (10, LOW);
  }

  else if (data == '2') {
    digitalWrite (LED_BUILTIN, LOW);
    digitalWrite (12, HIGH);
    digitalWrite (11, LOW);
    digitalWrite (10, LOW);
  }
    

  else if (data == '3') {
    digitalWrite (LED_BUILTIN, LOW);
    digitalWrite (12, LOW);
    digitalWrite (11, HIGH);
    digitalWrite (10, LOW);
  }
  

  else if (data == '4') {
    digitalWrite (LED_BUILTIN, LOW);
    digitalWrite (12, LOW);
    digitalWrite (11, LOW);
    digitalWrite (10, HIGH);
  }
    

  else if (data == '5') {
    digitalWrite (LED_BUILTIN, HIGH);
    digitalWrite (12, LOW);
    digitalWrite (11, HIGH);
    digitalWrite (10, LOW);
  }
    

  else if (data == '6') {
    digitalWrite (LED_BUILTIN, HIGH);
    digitalWrite (12, LOW);
    digitalWrite (11, LOW);
    digitalWrite (10, HIGH);
  }
    

  else if (data == '7') {
    digitalWrite (LED_BUILTIN, LOW);
    digitalWrite (12, HIGH);
    digitalWrite (11, HIGH);
    digitalWrite (10, LOW);
  }
  

  else if (data == '8') {
    digitalWrite (LED_BUILTIN, LOW);
    digitalWrite (12, HIGH);
    digitalWrite (11, LOW);
    digitalWrite (10, HIGH);
  }
    

  else if (data == '9') {
    digitalWrite (LED_BUILTIN, LOW);
    digitalWrite (12, LOW);
    digitalWrite (11, LOW);
    digitalWrite (10, LOW);
  }
  

}
