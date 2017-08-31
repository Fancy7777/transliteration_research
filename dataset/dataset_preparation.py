import sys

def process(fname):
    data =[]
    try:
        with open(fname) as f:
            for line in f:
        		#use gb18030 because is the origin enocode gb2312 is too small
        		#guess the reason is that they upload several times
        		#they forget to tell people gb2312 is not large enough
                data.append(line.decode('gb18030').encode("utf-8"))
    #there are still some encode problem but for some dataset
    #but they are not many
    #the origin gb2312 will lose 2/3 data
    except UnicodeDecodeError:
        pass

    #write to file and change the name to fname_transformed.txt	
    write_file = open(fname.split('.txt')[0]+"_transformed"+".txt",'w')
    for ele in data:
        write_file.write(ele)
    f.close()

if __name__ == "__main__":
	fname = sys.argv[1]
	process(fname)