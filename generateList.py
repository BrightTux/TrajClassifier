f= open("test.csv","w+")

startNum = 1
endNum = 44
for i in range(startNum, endNum+1):
     f.write("%d;NoActivity\r\n" % (i))
     i += 1

startNum = 45
endNum = 97
for i in range(startNum, endNum+1):
     f.write("%d;Traj1\r\n" % (i))
     i += 1

startNum = 98
endNum = 117
for i in range(startNum, endNum+1):
     f.write("%d;Traj2\r\n" % (i))
     i += 1

startNum = 118
endNum = 167
for i in range(startNum, endNum+1):
     f.write("%d;Traj3\r\n" % (i))
     i += 1

startNum = 168
endNum = 212
for i in range(startNum, endNum+1):
     f.write("%d;Traj5\r\n" % (i))
     i += 1

startNum = 213
endNum = 256
for i in range(startNum, endNum+1):
     f.write("%d;Traj6\r\n" % (i))
     i += 1

startNum = 257
endNum = 286
for i in range(startNum, endNum+1):
     f.write("%d;Traj7\r\n" % (i))
     i += 1

startNum = 287
endNum = 305
for i in range(startNum, endNum+1):
     f.write("%d;Traj8\r\n" % (i))
     i += 1

startNum = 306
endNum = 329
for i in range(startNum, endNum+1):
     f.write("%d;Traj9\r\n" % (i))
     i += 1

f.close()
