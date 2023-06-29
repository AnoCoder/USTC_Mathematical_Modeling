import re
import numpy as np
from PIL import Image

'''
手动实现的huffman压缩算法，运行时间稍长
因中间生成的huffman编码占用存储空间较大(43.7MB)，故不对编码结果写入文件，直接将其解码
'''


def combine_nodes(nodes):
    pos = 0
    newnode = []
    if len(nodes) > 1:
        nodes.sort()
        nodes[pos].append("1")                       # assigning values 1 and 0
        nodes[pos+1].append("0")
        combined_node1 = (nodes[pos][0] + nodes[pos+1][0])
        combined_node2 = (nodes[pos][1] + nodes[pos+1][1])  # combining the nodes to generate pathways
        newnode.append(combined_node1)
        newnode.append(combined_node2)
        newnodes = []
        newnodes.append(newnode)
        newnodes = newnodes + nodes[2:]
        nodes = newnodes
        huffman_tree.append(nodes)
        combine_nodes(nodes)
    return huffman_tree                                     # huffman tree generation


def huffman(file):
    print("Huffman Compression Program")
    print("=================================================================")
    my_string = np.asarray(Image.open(file), np.uint8)
    shape = my_string.shape
    a = my_string
    my_string = str(my_string.tolist())

    letters = []
    only_letters = []
    for letter in my_string:
        if letter not in letters:
            frequency = my_string.count(letter)             # frequency of each letter repetition
            letters.append(frequency)
            letters.append(letter)
            only_letters.append(letter)

    nodes = []
    while len(letters) > 0:
        nodes.append(letters[0:2])
        letters = letters[2:]                               # sorting according to frequency
    nodes.sort()

    huffman_tree.append(nodes)                             # Make each unique character as a leaf node
    newnodes = combine_nodes(nodes)

    huffman_tree.sort(reverse=True)
    print("Huffman tree with merged pathways:")

    checklist = []
    for level in huffman_tree:
        for node in level:
            if node not in checklist:
                checklist.append(node)
            else:
                level.remove(node)
    count = 0
    for level in huffman_tree:
        print("Level", count, ":", level)    # print huffman tree
        count += 1
    print()

    letter_binary = []
    if len(only_letters) == 1:
        lettercode = [only_letters[0], "0"]
        letter_binary.append(lettercode*len(my_string))
    else:
        for letter in only_letters:
            code = ""
            for node in checklist:
                if len(node) > 2 and letter in node[1]:  # genrating binary code
                    code = code + node[2]
            lettercode = [letter, code]
            letter_binary.append(lettercode)

    bitstring = ""
    for character in my_string:
        for item in letter_binary:
            if character in item:
                bitstring = bitstring + item[1]
    binary = "0b"+bitstring
    uncompressed_file_size = len(my_string)*7
    compressed_file_size = len(binary)-2
    print("original image size was", uncompressed_file_size,
          "bits. The compressed size is:", compressed_file_size, "bits")
    print("compression ratio is", uncompressed_file_size/compressed_file_size)
    print("Decoding.......")

    bitstring = str(binary[2:])
    uncompressed_string = ""
    code = ""
    for digit in bitstring:
        code = code+digit
        pos = 0                                        # iterating and decoding
        for letter in letter_binary:
            if code == letter[1]:
                uncompressed_string = uncompressed_string+letter_binary[pos][0]
                code = ""
            pos += 1

    temp = re.findall(r'\d+', uncompressed_string)
    res = list(map(int, temp))
    res = np.array(res)
    res = res.astype(np.uint8)
    res = np.reshape(res, shape)
    print("Input image dimensions:", shape)
    print("Output image dimensions:", res.shape)
    data = Image.fromarray(res)
    data.save('decompressed.png')
    if a.all() == res.all():
        print("Success")


if __name__ == '__main__':
    huffman_tree = []
    # file = "../assets/image/image_bmp/lena.bmp"
    file = "../assets/image/image_bmp/sunrise.bmp"
    huffman(file)
