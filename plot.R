df <- data.frame(
  read.csv("S0.csv"),
  read.csv("S1.csv"),
  read.csv("S2.csv"),
  read.csv("S3.csv"),
  read.csv("S4.csv"),
  read.csv("S5.csv"),
  read.csv("S6.csv"),
  read.csv("S7.csv")
)
names(df) <- c("S0","S1","S2","S3","S4","S5","S6","S7")

plot.ts(df, nc=1)
abline(v = 1000, col="red")
abline(v = 2000, col="red")
abline(v = 3000, col="red")
