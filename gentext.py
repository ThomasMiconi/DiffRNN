import random
import sys
n = 0
s1 = ["a", "b", "b", "a", "a", "b"]
while True:
    s2 = s1[::-1]
    pos = random.randint(0, len(s2)-1)
    if s2[pos] == "a":
        s2[pos] = "b"
    else:
        s2[pos] = "a"
    s1 = s2
    n += len(s2)+2
    sys.stdout.write("("+"".join(s2)+")")
    if n > 1200000:
        break
#for n in range (200000):
#    sys.stdout.write("(ab")
#    while (random.random() < .6):
#        sys.stdout.write("ab")
#    sys.stdout.write(")")

#for n in range (200000):
#    if random.random() < .5:
#        sys.stdout.write("aa")
#    else:
#        sys.stdout.write("bb")
