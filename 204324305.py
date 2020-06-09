import os
import random
import math
from math import log as ln
from itertools import combinations
import copy


def main():
    points_rectangle = get_data_set()  # data set of rectangle
    points_circle = copy.deepcopy(points_rectangle)  # data set of circle
    print('-----------------rectangle----------------------')
    for r in range(1, 9):
        sum_of_correct_ans_test_rect = 0  # sum of the correct answers on the train of rectangle
        sum_of_correct_ans_train_rect = 0  # sum of the correct amswers on the test of rectangle
        for i in range(0, 100):
            random.shuffle(points_rectangle)  # shuffle the dataset
            H_rectangle_list = copy.deepcopy(adaboost(rectangle, points_rectangle, r))  # list of the rectagles from adaboost
            sum_of_correct_ans_train_rect += test_adaboost(H_rectangle_list, points_rectangle[0:65], 'rectangle')  # test the train and add the errors to sum
            sum_of_correct_ans_test_rect += test_adaboost(H_rectangle_list, points_rectangle[65: 130], 'rectangle')
        sum_of_correct_ans_train_rect = sum_of_correct_ans_train_rect/100  # divide the sum_err in 100
        sum_of_correct_ans_test_rect = sum_of_correct_ans_test_rect/100
        print('round ' + str(r)+' train result = ' + str(sum_of_correct_ans_train_rect/65))  # divide the errors in 65 for precent
        print('round ' + str(r)+' test result = ' + str(sum_of_correct_ans_test_rect/65))
        print('')
    print('------------------------circle---------------------')
    for r in range(1, 9):
        sum_correct_ans_circ_test = 0
        sum_correct_ans_circ_train = 0
        for i in range(0, 100):
            random.shuffle(points_circle)
            H_circle_list = list()
            H_circle_list = copy.deepcopy(adaboost(circle, points_circle, r))
            sum_correct_ans_circ_train += test_adaboost(H_circle_list, points_circle[0:65], 'circle')
            sum_correct_ans_circ_test += test_adaboost(H_circle_list, points_circle[65:130], 'circle')
        print('round ' + str(r)+' train result = ' + str(sum_correct_ans_circ_train/100))
        print('round ' + str(r)+' test result = ' + str(sum_correct_ans_circ_test/100))
        print('')


def get_data_set():
    f = open('HC_Body_Temperature.txt', "r")
    temp_points = list(f.read().splitlines())  # get the points from the file
    points = list()  # list of lists
    get_listof_points(points, temp_points)  # each point in points is a list that contain [x, y, label, wegiht]
    return points


def adaboost( func_shape, points, r):  # get a function (circle or rectangle), dataset and number of round
    for point in points[0:65]:  # intilize the first 65 points(train set) weight to 1/65
        point[3] = 1/65  # point[3] is wight of the point
    hi_list = list()  # list of the r rules
    for i in range(0, r):
        temp_hi = func_shape(points)  # call circle or rectangle and return (points of shape, error of the shape)
        in_label = temp_hi[3]  # the label that inside the shape (1 or -1)
        err_i = temp_hi[2]  # error of the shape
        alpha_i = (ln((1 - err_i) / err_i)) / 2
        hi = (temp_hi[0], temp_hi[1], alpha_i, temp_hi[3], temp_hi[4])  # hi is (point 1,point 2, alpha, label and id: circle or rectangle)
        hi_list.append(hi)
        if err_i >= 0.5:  # if err >= 0.5 then there is no point ot continue running on this round
            return hi_list
        compute_new_weights(points, alpha_i, hi, in_label)  # calculate the new weights
        normalize_wegihts(points)  # normalize the weights
    return hi_list


def test_adaboost(hi_arr, points, id):  # this function call the correct test function acording to its shape
    if id == 'rectangle':
       return test_rectangle(hi_arr, points)
    elif id == 'circle':
       return test_circle(hi_arr, points)


def test_circle(hi_arr, points):
    correct_classify = 0
    for point in points:
        sum_alpha = 0  # sum of the alpha
        label_point = point[2]  # label of the point
        classify = 0  # the class label that the rules will decide on the point
        for shape in hi_arr:
            alpha_shape_i = shape[2]  # alpha of the current circle
            if in_or_out_circle(shape[0], shape[1], point):  # if the point is inside the circle
                sum_alpha += alpha_shape_i * shape[3]
            elif not in_or_out_circle(shape[0], shape[1], point):  # if the point is outside the circle
                sum_alpha += alpha_shape_i * (-shape[3])
        if sum_alpha > 0:  # if sum_alpha is poisitive then classify the point as 1
            classify = 1
        else:
            classify = -1  # else classify it as -1
        if label_point == classify:  # if correct the add 1 to the correct answers
            correct_classify += 1
    return correct_classify/65


def test_rectangle(hi_arr, points):  # very similer to the test_circle
    correct_clasiffy_test = 0
    for point in points:
        sum_alpha = 0
        classify = 0
        label_point = point[2]
        for hi in hi_arr:
            alpha_hi = hi[2]
            max_x = max(hi[0][0], hi[1][0])
            max_y = max(hi[0][1], hi[1][1])
            min_y = min(hi[0][1], hi[1][1])
            min_x = min(hi[0][0], hi[1][0])
            if check_position_in_or_out(max_x, max_y, min_x, min_y, point):
                sum_alpha += alpha_hi * hi[3]
            else:
                 sum_alpha += alpha_hi * (- hi[3])

        if sum_alpha > 0:
            classify = 1
        else:
            classify = -1
        if label_point == classify:
            correct_clasiffy_test += 1
    return correct_clasiffy_test


def get_listof_points(points, temp_points):
    for point in temp_points:
        temp = point.split()  # get the point x y and label without spaces
        x = float(temp[0])  # turn the x to float
        y = int(temp[2])  # turn y to int
        label = int(temp[1])  # turn label to int
        weight = 1 / 65
        number_point = [x, y, label, weight]
        tuple_point = list(number_point)  # build a tuple of (x, y, label)
        points.append(tuple_point)  # add the tuple to the tuple list


def rectangle(points):
    comb = combinations(points[0:65], 2)  # all possible combinations of pair of points
    rectangle_err_list = list()  # list of the errors and rectangles
    for rect in list(comb):
        max_min_tuple = get_max_x_and_max_y(rect)  # get the max and min y and x from the two points
        max_x = max_min_tuple[0]
        min_x = max_min_tuple[1]
        max_y = max_min_tuple[2]
        min_y = max_min_tuple[3]
        point_1 = (rect[0][0],rect[0][1])  # first point of the rectangle
        point_2 = (rect[1][0],rect[1][1])  # second point of the rectangle
        sumof_positive_err = 0
        sumof_negetive_err = 0
        for i in range(1, 3):
            for point in points[0:65]:  # first 65 points are the train set
                weight_point = point[3]  # point is:(x,y,label,weight)
                if i == 1:
                    if check_position_in_or_out(max_x, max_y,min_x,min_y,point):  # if point is inside the rectangle
                        if point[2] != 1:  # if the label of point is not one then add 1 to the poisive rectangle errors
                            sumof_positive_err += weight_point
                    else:  # if the point is outside the rectangle
                        if point[2] == 1:
                            sumof_positive_err += weight_point
                elif i == 2:  # check for rectangle of -1
                    if check_position_in_or_out(max_x, max_y, min_x, min_y, point):
                        if point[2] != -1:
                            sumof_negetive_err += weight_point
                    else:
                        if point[2] == -1:
                            sumof_negetive_err += weight_point

        if sumof_negetive_err < sumof_positive_err:
            rectangle_err_list.append((point_1, point_2, sumof_negetive_err, -1, 'rectangle'))
        else:
            rectangle_err_list.append((point_1, point_2, sumof_positive_err, 1, 'rectangle'))  # add the rectangle to the list
    return find_minimum_error_rectangle(rectangle_err_list)  # return the rectangle with the minimum error


def check_position_in_or_out( max_x, max_y, min_x, min_y,cur_point):  # return true if the point is inside the rectangle
    cur_x_point = cur_point[0]
    cur_y_point = cur_point[1]
    if max_x == min_x:  # if the points of the rectangle have the same x then make min_x zero
        min_x = 0
    if max_y == min_y:  # same as above
        min_y = 0
    if ((cur_x_point >= min_x) and (cur_x_point <= max_x)) and ((cur_y_point >= min_y) and (cur_y_point <= max_y)):
        return True
    else:
        return False



def get_max_x_and_max_y(point):  # return a tuple of maximum and minimum x and y for rectangle
    x_1 = point[0][0]
    x_2 = point[1][0]
    y_1 = point[0][1]
    y_2 = point[1][1]
    max_x = max(x_1, x_2)
    min_x = min(x_1, x_2)
    max_y = max(y_1, y_2)
    min_y = min(y_1, y_2)
    return max_x, min_x, max_y, min_y


def find_minimum_error_rectangle(rectangles):  # find the rectangle with the minimum error
    min_err = rectangles[0][2]
    best_rectangle = copy.deepcopy(rectangles[0])
    for rect in rectangles:
        curr_err = rect[2]
        if curr_err < min_err:
            min_err = curr_err
            best_rectangle = copy.deepcopy(rect)
    return best_rectangle


def compute_new_weights(points, alpha_t, hi, in_label):  # update the weights
    if hi[4] == 'rectangle':  # if hi is rectangle
        hi_point_1 = hi[0]  # point 1 of hi
        hi_point_2 = hi[1]  # point 2 of hi
        max_x = max(hi_point_1[0], hi_point_2[0])
        min_x = min(hi_point_1[0], hi_point_2[0])
        max_y = max(hi_point_1[1], hi_point_2[1])
        min_y = min(hi_point_1[1], hi_point_2[1])
        for i in range(0, 65):
            temp_wegiht = points[i][3]  # weight of the point
            curr_point = points[i]
            if check_position_in_or_out(max_x, max_y, min_x, min_y, curr_point):  # if classify right
                if points[i][2] == in_label:
                    points[i][3] = temp_wegiht * math.exp(-alpha_t)
                elif points[i][2] != in_label:
                    points[i][3] = temp_wegiht * math.exp(alpha_t)
            else:
                if points[i][2] != in_label:
                    points[i][3] = temp_wegiht * math.exp(-alpha_t)  # if error on classify
                elif points[i][2] == in_label:
                    points[i][3] = temp_wegiht * math.exp(alpha_t)  # if error on classify
    elif hi[4] == 'circle':  # if hi is circle
        center_point = hi[0]  # center of hi
        round_point = hi[1]
        for i in range(0, 65):
            point_wegiht = points[i][3]  # weight of the point
            curr_point = points[i]  # the current point
            if in_or_out_circle(center_point, round_point, curr_point):  # if the point inside the circle
                if curr_point[2] == in_label:  # if the point label is equal to the label of the circle
                    points[i][3] = point_wegiht * math.exp(-1 * alpha_t)
                else:
                    points[i][3] = point_wegiht * math.exp(alpha_t)
            elif not in_or_out_circle(center_point, round_point, curr_point):
                if curr_point[2] != in_label:
                    points[i][3] = point_wegiht * math.exp(-1 * alpha_t)
                else:
                    points[i][3] = point_wegiht * math.exp(alpha_t)


def normalize_wegihts(points):
    sumof_wegihts = 0
    for point in points[0:65]:
        sumof_wegihts += point[3]
    for point in points[0:65]:
        point[3] = point[3] / sumof_wegihts


def circle(points):
    comb = combinations(points[0:65], 2)
    circle_list = list()  # list of all circles
    for circ in list(comb):
        for j in range(0, 2):  # run twice on eache pair of points: one time point1 is the center and next point2 is the center
            if j == 0:
                center_point = (circ[0][0], circ[0][1])
                round_point = (circ[1][0], circ[1][1])
            else:
                center_point = (circ[1][0], circ[1][1])
                round_point = (circ[0][0], circ[0][1])
            sumof_positive_err = 0
            sumof_negetive_err = 0
            for i in range(0, 2):
                for point in points[0:65]:  # run on the first 65 points(trainset)
                    weight_point = point[3]
                    if i == 0:  # circle label is 1
                        if in_or_out_circle(center_point, round_point, point):  # if the point is inside the circle
                            if point[2] == -1:
                                sumof_positive_err += weight_point
                        else:
                            if point[2] == 1:
                                sumof_positive_err += weight_point
                    else:  # circle label is -1
                        if in_or_out_circle(center_point, round_point, point):
                            if point[2] == 1:
                                sumof_negetive_err += weight_point
                        else:
                            if point[2] == -1:
                                sumof_negetive_err += weight_point
            if sumof_negetive_err < sumof_positive_err:
                circle_list.append((center_point, round_point, sumof_negetive_err, -1, 'circle'))
            else:
                circle_list.append((center_point, round_point, sumof_positive_err, 1, 'circle'))
    return find_best_circle(circle_list)  # return the best circle


def in_or_out_circle(center, round_point, curr_point):
    x_center = center[0]
    y_center = center[1]
    x_round = round_point[0]
    y_round = round_point[1]
    radius = math.sqrt(math.pow(x_center - x_round, 2) + math.pow(y_center - y_round, 2))  # radius of the circle
    dist = math.sqrt(math.pow(x_center - curr_point[0], 2) + math.pow(y_center - curr_point[1], 2))  # distance of the current point from the circle
    if dist <= radius:  # if distance <= radius, the the point is inside the circle
        return True
    else:
        return False


def find_best_circle(circle_list):  # return the circle with the minimum error
    min_err = circle_list[0][2]
    best_circ = copy.deepcopy(circle_list[0])
    for circle in circle_list:
        err_circ = circle[2]
        if err_circ < min_err:
            min_err = err_circ
            best_circ = copy.deepcopy(circle)
    return best_circ


main()



