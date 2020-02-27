import numpy as np
from PIL import Image, ImageDraw


def main():
    img = Image.open('exm.JPG')
    img_gray = img.convert('L')  # converts the image to grayscale image
    # img_bin = img.convert('1') #converts to a binary image, T=128, LOW=0, HIGH=255
    img_gray.show()
    ONE = 150
    a = np.asarray(img_gray)  # from PIL to np array
    a_bin = threshold(a, 100, ONE, 0)
    im = Image.fromarray(a_bin)  # from np array to PIL format
    im.show()

    label = blob_coloring_8_connected(a_bin, ONE)
    new_img2 = np2PIL_color(label)
    new_img2.show()

    rectangle_array = rectangle(label, img)

    resize(img, rectangle_array)

    # example
    # a_bin = binary_image(100,100, ONE)   #creates a binary image
    # label = blob_coloring_4_connected(a_bin, ONE)
    # new_img2 = np2PIL_color(label)
    # new_img2.show()


def binary_image(nrow, ncol, Value):
    x, y = np.indices((nrow, ncol))
    mask_lines = np.zeros(shape=(nrow, ncol))

    x0, y0, r0 = 30, 30, 10
    x1, y1, r1 = 70, 30, 10

    for i in range(50, 70):
        mask_lines[i][i] = 1
        mask_lines[i][i + 1] = 1
        mask_lines[i][i + 2] = 1
        mask_lines[i][i + 3] = 1
        mask_lines[i][i + 6] = 1
        mask_lines[i - 20][90 - i + 1] = 1
        mask_lines[i - 20][90 - i + 2] = 1
        mask_lines[i - 20][90 - i + 3] = 1

    # mask_circle1 = np.abs((x - x0) ** 2 + (y - y0) ** 2 - r0 ** 2 ) <= 5
    mask_square1 = np.fmax(np.absolute(x - x1), np.absolute(y - y1)) <= r1
    # mask_square2 = np.fmax(np.absolute( x - x2), np.absolute( y - y2)) <= r2
    # mask_square3 = np.fmax(np.absolute( x - x3), np.absolute( y - y3)) <= r3
    # mask_square4 =  np.fmax(np.absolute( x - x4), np.absolute( y - y4)) <= r4
    # imge = np.logical_or ( np.logical_or(mask_lines, mask_circle1), mask_square1) * Value
    imge = np.logical_or(mask_lines, mask_square1) * Value
    # imge = np.logical_or(mask_lines, mask_circle1) * Value

    return imge


def np2PIL(im):
    print("size of arr: ", im.shape)
    img = Image.fromarray(im, 'RGB')
    return img


def np2PIL_color(im):
    print("size of arr: ", im.shape)
    img = Image.fromarray(np.uint8(im))
    return img


def threshold(im, T, LOW, HIGH):
    (nrows, ncols) = im.shape
    im_out = np.zeros(shape=im.shape)
    for i in range(nrows):
        for j in range(ncols):
            if abs(im[i][j]) < T:
                im_out[i][j] = LOW
            else:
                im_out[i][j] = HIGH
    return im_out


def blob_coloring_8_connected(bim, ONE):
    max_label = int(10000)
    nrow = bim.shape[0]
    ncol = bim.shape[1]
    print("nrow, ncol", nrow, ncol)
    im = np.zeros(shape=(nrow, ncol), dtype=int)
    a = np.zeros(shape=max_label, dtype=int)
    a = np.arange(0, max_label, dtype=int)
    color_map = np.zeros(shape=(max_label, 3), dtype=np.uint8)
    color_im = np.zeros(shape=(nrow, ncol, 3), dtype=np.uint8)

    for i in range(max_label):
        np.random.seed(i)
        color_map[i][0] = np.random.randint(0, 255, 1, dtype=np.uint8)
        color_map[i][1] = np.random.randint(0, 255, 1, dtype=np.uint8)
        color_map[i][2] = np.random.randint(0, 255, 1, dtype=np.uint8)

    k = 0
    for i in range(nrow):
        for j in range(ncol):
            im[i][j] = max_label
    for i in range(1, nrow - 1):
        for j in range(1, ncol - 1):
            c = bim[i][j]
            l = bim[i][j - 1]
            u = bim[i - 1][j]
            d = bim[i - 1][j - 1]
            r = bim[i - 1][j + 1]

            label_u = im[i - 1][j]
            label_l = im[i][j - 1]
            label_d = im[i - 1][j - 1]
            label_r = im[i - 1][j + 1]

            im[i][j] = max_label
            if c == ONE:
                min_label = min(label_u, label_l, label_d, label_r)
                if min_label == max_label:
                    k += 1
                    im[i][j] = k
                else:
                    im[i][j] = min_label
                    if min_label != label_u and label_u != max_label:
                        update_array(a, min_label, label_u)

                    if min_label != label_l and label_l != max_label:
                        update_array(a, min_label, label_l)

                    if min_label != label_d and label_d != max_label:
                        update_array(a, min_label, label_d)

                    if min_label != label_r and label_r != max_label:
                        update_array(a, min_label, label_r)

            else:
                im[i][j] = max_label
    # final reduction in label array
    for i in range(k + 1):
        index = i
        while a[index] != index:
            index = a[index]
        a[i] = a[index]

    # second pass to resolve labels and show label colors
    for i in range(nrow):
        for j in range(ncol):

            if bim[i][j] == ONE:
                im[i][j] = a[im[i][j]]
                if im[i][j] == max_label:
                    im[i][j] == 0
                    color_im[i][j][0] = 0
                    color_im[i][j][1] = 0
                    color_im[i][j][2] = 0
                color_im[i][j][0] = color_map[im[i][j], 0]
                color_im[i][j][1] = color_map[im[i][j], 1]
                color_im[i][j][2] = color_map[im[i][j], 2]
    return color_im


def update_array(a, label1, label2):
    index = lab_small = lab_large = 0
    if label1 < label2:
        lab_small = label1
        lab_large = label2
    else:
        lab_small = label2
        lab_large = label1
    index = lab_large
    while index > 1 and a[index] != lab_small:
        if a[index] < lab_small:
            temp = index
            index = lab_small
            lab_small = a[temp]
        elif a[index] > lab_small:
            temp = a[index]
            a[index] = lab_small
            index = temp
        else:  # a[index] == lab_small
            break

    return


def convert(set):
    return list(set)


def rectangle(label, img):
    nrow = label.shape[0]
    ncol = label.shape[1]

    # finding rgb values
    rgb = set(tuple(v) for item in label for v in item)

    # converting set with rgb values into a list
    rgb_list = convert(rgb)

    k = len(rgb)

    # finding coordinates of characters
    rectangle_array = np.zeros((k, 4))

    for i in range(k):
        rectangle_array[i][0] = 10000
        rectangle_array[i][1] = 10000

    for i in range(nrow):
        for j in range(ncol):
            # checking if current label has rgb values
            if label[i][j][0] != 0 and label[i][j][1] != 0 and label[i][j][2] != 0:
                # detecting current index
                current_R = label[i][j][0]
                current_G = label[i][j][1]
                current_B = label[i][j][2]
                index = -1
                for z in range(k):
                    if current_R == rgb_list[z][0] and current_G == rgb_list[z][1] and current_B == rgb_list[z][2]:
                        index = z
                        break

                # index is known from this point
                for t in range(k):
                    # finding min_i
                    if i < rectangle_array[index][0]:
                        rectangle_array[index][0] = i
                    # finding min_j
                    if j < rectangle_array[index][1]:
                        rectangle_array[index][1] = j
                    # finding max_i
                    if i > rectangle_array[index][2]:
                        rectangle_array[index][2] = i
                    # finding max_j
                    if j > rectangle_array[index][3]:
                        rectangle_array[index][3] = j

    # drawing the rectangles
    for i in range(k):
        shape = [(rectangle_array[i][1], rectangle_array[i][0]), (rectangle_array[i][3], rectangle_array[i][2])]
        img1 = ImageDraw.Draw(img)
        img1.rectangle(shape, fill=None, outline="red")
    img.show()

    return rectangle_array


def resize(img, rectangle_array):
    for i in range(len(rectangle_array)):
        # setting the points for cropped image
        left = rectangle_array[i][1]
        top = rectangle_array[i][0]
        right = rectangle_array[i][3]
        bottom = rectangle_array[i][2]

        # cropped image of above dimension
        img1 = img.crop((left, top, right, bottom))
        new_size = (21, 21)
        img1 = img1.resize(new_size)
        img1.show()


if __name__ == '__main__':
    main()
