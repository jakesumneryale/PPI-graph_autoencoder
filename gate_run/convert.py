def convert(s, numRows):

    if len(s) == 1:
        return s

    ## Do some calculations for everything

    mid_rows = numRows - 2

    ## First row

    first_row = ""

    first_row_factor = 2 * (numRows - 1)

    for i in range((len(s)//first_row_factor)+1):
        first_row += s[first_row_factor*i]

    ## Last row

    last_row = ""

    for i in range((len(s)//first_row_factor)+1):
        last_ind = first_row_factor*i + numRows - 1

        if last_ind < len(s):
            last_row += s[last_ind]


    ## Middle rows

    mid_rows_str = ""

    for i in range(mid_rows):
        for j in range((len(s)//first_row_factor)+1):

            ## in col
            mid_col_ind = (j*first_row_factor) + (i+1)
            if mid_col_ind < len(s):
                mid_rows_str += s[mid_col_ind]

            ## in diag

            diag_ind = (first_row_factor*j + i + 1) + (numRows -(2+i)) * 2

            if diag_ind < len(s):
                mid_rows_str += s[diag_ind]

    # print(first_row)
    # print(mid_rows_str)
    # print(last_row)

    return first_row + mid_rows_str + last_row



def main():
    s = "PAYPALISHIRING"
    numRows = 4
    print(convert(s, numRows))
    print("PINALSIGYAHRPI")
    print("PINALSIGYAHRPI" == convert(s, numRows))

    numRows = 3

    print(convert(s,numRows))
    print("PAHNAPLSIIGYIR")
    print("PAHNAPLSIIGYIR" == convert(s, numRows))

main()