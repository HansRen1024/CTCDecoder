def fast_ctc_decode(char_num,ind):
'''
char_num is the output of CRNN network.
ind is the index in a batch samples.
'''
    def MaxProbability(row, MaxIndex, curPro):
        maxPro = curPro
        maxRow = row
        if row != 0:
            num = 0
            while 1:
                num += 1
                LaMaxIndex = char_num[ind, row - num, :].tolist().index(max(char_num[ind, row - num, :].tolist())) # go up to find max probability in row-num
                if MaxIndex != LaMaxIndex: break # if find another label, break
                if MaxIndex==LaMaxIndex and max(char_num[ind, row-num, :].tolist())>maxPro: # upgrade max probability and its row
                    maxPro = max(char_num[ind, row-num, :].tolist())
                    maxRow = row-num
        return maxPro, maxRow
    ResultList = []  # format： [index_with_max_prob，max_prob, row_of_max_prob]
    for row in range(0, char_num[ind, :, :].shape[0]):
        MaxIndex = char_num[ind, row, :].tolist().index(max(char_num[ind, row, :].tolist())) # max probability index in current row
        if row == char_num[ind, :, :].shape[0] - 1: # last row
            if MaxIndex != (char_num[ind, :, :].shape[1]-1): # not blank label
                maxPro, maxRow = MaxProbability(row, MaxIndex, max(char_num[ind, row, :].tolist())) # find max probability and its row
                ResultList.append([MaxIndex, maxPro, maxRow])
            continue
        NeMaxIndex = char_num[ind, row + 1, :].tolist().index(max(char_num[ind, row + 1, :].tolist())) # next row
        if NeMaxIndex != MaxIndex and MaxIndex != (char_num[ind, :, :].shape[1]-1): # current lable not equals to next lable and neither a blank
            maxPro, maxRow = MaxProbability(row, MaxIndex, max(char_num[ind, row, :].tolist()))
            ResultList.append([MaxIndex, maxPro, maxRow])
    return ResultList
