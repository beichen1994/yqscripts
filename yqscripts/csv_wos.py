import sys
import os
import csv
import argparse

"""
    Arguments:
    
    `-h`: help file of usage of the script
    `-i`: input file name
    `-o`: output file name

"""


def get_arguments():

    # Use arparse to get command line arguments
    parser = argparse.ArgumentParser('convert csv')
    parser.add_argument("-i", "--input_file", required=True,
                        help="csv input file (with extension)", type=str)
    parser.add_argument("-o", "--output_file", required=True,
                        help="csv output file (without extension)", type=str)
    args = parser.parse_args()

    # Check if the input_file exits
    is_valid_file(parser, args.input_file)

    return args.input_file, args.output_file


def is_valid_file(parser, file_name):
    """Ensure that the input_file exists."""
    if not os.path.exists(file_name):
        parser.error("The file '{}' does not exist!".format(file_name))
        sys.exit(1)


def parse_file(arguments):
    input_file = arguments[0]
    output_file = arguments[1]
    output_path = '.'  # Current directory
    csv_df = pd.read_csv(input_file,encoding='utf-8',index=False) 
    keywords = list(csv_df['keywords'].values) #csv文件中必须有keywords这一列，并且里面的关键词必须是用空格分割的
    year_ = list(csv_df['time'].values) #csv文件中必须有time这一列，并且里面的时间为年份
    keywords_list=[]
    for i in keywords:
        keywords_str=''
        for j in str(i).split(' '):
            if j != ' ':
                keywords_str = keywords_str+j+'; '
        keywords_str = keywords_str.strip('; ; ')
        keywords_list.append(keywords_str)
    rows = len(keywords_list)
    domain_dfs = pd.DataFrame({'DE':keywords_list})
    domain_dfs['UT']=[i for i in range(rows)]
    domain_dfs['PY'] = year_ 
    order=['UT','PY','DE']
    domain_dfs=domain_dfs[order]
    UT = list(domain_dfs.UT.values)
    PY = list(domain_dfs.PY.values)
    DE = list(domain_dfs.DE.values)
    current_output = os.path.join(output_path,"{}".format(output_file))
    f = open(current_output,'w',encoding='utf-8')
    f.write('UT	PY	DE')
    f.write('\n')
    for i in range(domain_dfs.shape[0]):
        f.write(str(UT[i]))
        f.write('	')
        f.write(str(PY[i]))
        f.write('	')
        f.write(str(DE[i]))
        f.write('\n')
    f.close()


def main():
    arguments = get_arguments()
    parse_file(arguments)

if __name__ == "__main__":
    main()

