def histogram_filter(p, field, measurement, motion, sensorTrust, actionTrust):
    """
    Histogram filter implementation for SLAG.
    Use 20,000 bins: 20 bins for x_r * 10 bins for x_l * 10 bins for y_l * 10 bins for x_theta
    move function takes in robot arm motion, returns p_bar for all bins
    sense function takes in the beam measurement, returns p for all bins (final probability of some bin being true)
    """
    def normalizer(q, sum, colsNumber):
        normalized = []
        row = []
        for i in range(len(q)):
            q[i] = q[i] / sum
            row.append(q[i])
            if (i + 1) % colsNumber == 0:
                normalized.append(row)
                row = []
        return normalized

    def move(p, motion, trust):
        q = []
        for row in range(len(p)):
            for col in range(len(p[row])):
                s = p[(row - motion[0]) % len(p)][(col - motion[1]) % len(p[row])] * trust
                s += p[row % len(p)][col % len(p[row])] * (1 - trust)
                q.append(s)

        s = sum(q)
        q = normalizer(q, s, len(p[0]))
        return q

    def sense(p, measurement, trust):
        q = []

        def xor(measurement, field, pZGivenX, fieldRow, fieldCol):
            outValue = 'B'
            for row in range(len(measurement)):
                for col in range(len(measurement[0])):
                    outscenario = (row - int(len(measurement) / 2)) + fieldRow < 0 or (
                                col - int(len(measurement[0]) / 2)) + fieldCol < 0 or (
                                              row - int(len(measurement) / 2)) + fieldRow >= B or (
                                              col - int(len(measurement[0]) / 2)) + fieldCol >= A
                    if outscenario:
                        if measurement[row][col] == outValue:
                            pZGivenX *= trust
                        else:
                            pZGivenX *= (1 - trust)

                    elif measurement[row][col] == field[fieldRow + row - int(len(measurement) / 2)][
                        fieldCol + col - int(len(measurement[0]) / 2)]:
                        pZGivenX *= trust
                    else:
                        pZGivenX *= (1 - trust)

            return pZGivenX

        for row in range(len(p)):
            for col in range(len(p[0])):
                pZGivenX = xor(measurement, field, 1, row, col)
                q.append(p[row][col] * pZGivenX)

        s = sum(q)
        q = normalizer(q, s, len(p[0]))
        return q

    p = move(p, motion, actionTrust)
    p = sense(p, measurement, sensorTrust)
    return p
