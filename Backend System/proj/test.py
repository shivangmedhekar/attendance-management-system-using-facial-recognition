import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd=""
)

mycursor = mydb.cursor()

mycursor.execute("use testproj")
mycursor.execute("SELECT * FROM test1")

myresult = mycursor.fetchall()

for x in myresult:
  print(x)