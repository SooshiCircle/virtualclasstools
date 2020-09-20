def checkForHandRaised(coords):
    if len(coords) >= 90:
        l = coords[:-90]
        x = [i[0] for i in l]
        y = [i[1] for i in l]

        avgx = sum(x)/len(x)
        avgy = sum(y)/len(y)

        good = 0
        for i in range(90):
            if abs(x[i] - avgx) < 15 and abs(y[i] - avgy) < 15:
                good += 1
        if good > 80:
            return True
        else:
            return False

