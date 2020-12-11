import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# numeric sudoku solver
from sudokusolver import solve


DIGIT_CLASSIFIER_PATH = '/autograder/submission/mnist.model'
# DIGIT_CLASSIFIER_PATH = './mnist.model'
SAMPLE_PATH = '/autograder/submission/train_7.jpg'
# SAMPLE_PATH = './train_7.jpg'
MNIST_CELL_SIZE = 28
NORMAL_TABLE_SIZE = 1000
BORDER_SIZE = 50


class MnistClassifier(nn.Module):
    def __init__(self):
        super(MnistClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, 
                               kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, 
                               kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=640, out_features=200)
        self.fc2 = nn.Linear(in_features=200, out_features=10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_digit_images(filepath):
    global DIGITS
    digit_model = MnistClassifier()
    digit_model.load_state_dict(torch.load(DIGIT_CLASSIFIER_PATH))
    digit_model.double()
    digit_model.eval()
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, tables, _ = search_tables(img)
    _, DIGITS = search_digits(tables[0], digit_model, return_digit_images=True)
    return 


DIGITS = {}
load_digit_images(SAMPLE_PATH)



def search_tables(img):
    # binarize image
    img = cv2.inRange(img, 0, 100)
    img = cv2.dilate(img, np.ones((3,3), np.uint8), iterations=1)

    # find big contours and filter them to leave sudoku only
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    max_contours_idx = list(np.argsort(areas)[-5:])
    sudoku_tables = []
    for cnt_idx in max_contours_idx:
        n_inside_contours = len(np.argwhere(hierarchy[0].T[3] == cnt_idx).reshape(-1))
        if n_inside_contours >= 81 - 3 and n_inside_contours <= 81 + 3:
            # filter points in sudoku tables to leave only 4 (?) that define a table
            epsilon = 0.01 * cv2.arcLength(contours[cnt_idx], True) 
            approx_table = cv2.approxPolyDP(contours[cnt_idx], epsilon, True)
            if (len(approx_table) >= 4) and (len(approx_table) <= 8):
                sudoku_tables.append(approx_table)
    
    # try to find a sudoku table once again
    if len(sudoku_tables) == 0:
        k = 1
        while k < 10:
            cnt_idx = np.argsort(areas)[-k]
            epsilon = 0.01 * cv2.arcLength(contours[cnt_idx], True) 
            approx_table = cv2.approxPolyDP(contours[cnt_idx], epsilon, True)
            k += 1
            if (len(approx_table) >= 4) and (len(approx_table) <= 20):
                sudoku_tables.append(approx_table)
                break
    
    # draw final binary mask: 1 for sudoku tables, 0 otherwise
    mask = np.zeros(img.shape)
    cv2.fillPoly(mask, sudoku_tables, color=(1, 1, 1))
    
    # rotate sudoku table to normal square table in points p1 -> (0, 0), p2 -> (R, 0), p3 -> (R, R), p4 -> (0, R)
    # derive points in order
    for idx, table in enumerate(sudoku_tables):
        center = table.mean(axis=0)[0]
        p1, p2, p3, p4 = [p[0] for p in table[np.random.choice(len(table), 4, replace=False)]]
        for p in table:
            if p[0][0] < center[0] and p[0][1] < center[1]:
                if p[0][0] + p[0][1] < p1[0] + p1[1]:
                    p1 = p[0]
            elif p[0][0] < center[0] and p[0][1] > center[1]:
                p2 = p[0]
            elif p[0][0] > center[0] and p[0][1] < center[1]:
                p3 = p[0]
            elif p[0][0] > center[0] and p[0][1] > center[1]:
                if p[0][0] + p[0][1] > p4[0] + p4[1]:
                    p4 = p[0]
            else:
                pass
        p = [p1, p2, p3, p4]
        sudoku_tables[idx] = p
    
    # final transform
    normal_points = np.float32([[0, 0],
                                [0, NORMAL_TABLE_SIZE],
                                [NORMAL_TABLE_SIZE, 0], 
                                [NORMAL_TABLE_SIZE, NORMAL_TABLE_SIZE]])
    for k in range(len(normal_points)):
        normal_points[k] = [normal_points[k][0] + BORDER_SIZE, normal_points[k][1] + BORDER_SIZE]
    normal_tables = []
    for table in sudoku_tables:
        transform = cv2.getPerspectiveTransform(np.float32(table), normal_points)
        normal_table = cv2.warpPerspective(img, transform, 
                                           (NORMAL_TABLE_SIZE + 2 * BORDER_SIZE, NORMAL_TABLE_SIZE + 2 * BORDER_SIZE))
        normal_tables.append(normal_table)

    return mask, normal_tables, sudoku_tables


def search_digits_bad_table(img_table):
    size = img_table.shape[0] - 2 * BORDER_SIZE
    h = size // 10
    step = size // 100
    digit_table = [[None for _ in range(9)] for _ in range(9)]
    null_pad = 2
    null_border = np.pad(np.ones((MNIST_CELL_SIZE - 2 * null_pad, MNIST_CELL_SIZE - 2 * null_pad)), pad_width=null_pad)
    null_pad_cell = 10
    null_border_cell_10 = np.pad(np.ones((h - 2 * null_pad_cell, h - 2 * null_pad_cell)), pad_width=null_pad_cell)
    for i in range(9):
        for j in range(9):
            img_cell = img_table[step * i + h * i + BORDER_SIZE + step // 2 : step * i + h * (i + 1) + BORDER_SIZE + step // 2,
                                 step * j + h * j + BORDER_SIZE + step // 2 : step * j + h * (j + 1) + BORDER_SIZE + step // 2]
            img_cell = np.pad(img_cell * null_border_cell_10, pad_width=3)
            img_cell = cv2.resize(img_cell, (MNIST_CELL_SIZE, MNIST_CELL_SIZE), interpolation=cv2.INTER_AREA)
            digit_table[i][j] = img_cell * null_border   
    return digit_table
 
            
def search_digits_good_table(img_table, cells):
    size = img_table.shape[0] - 2 * BORDER_SIZE
    moments = [cv2.moments(cnt) for cnt in cells]
    cx = [int(m['m10'] / m['m00']) for m in moments]
    cy = [int(m['m01'] / m['m00']) for m in moments]
    h = size // 10
    digit_table = [[None for _ in range(9)] for _ in range(9)]
    null_pad = 2
    null_border = np.pad(np.ones((MNIST_CELL_SIZE - 2 * null_pad, MNIST_CELL_SIZE - 2 * null_pad)), pad_width=null_pad)
    null_pad_cell = 10
    null_border_cell_10 = np.pad(np.ones((h - 2 * null_pad_cell, h - 2 * null_pad_cell)), pad_width=null_pad_cell)
    null_pad_cell = 20
    null_border_cell_20 = np.pad(np.ones((h - 2 * null_pad_cell, h - 2 * null_pad_cell)), pad_width=null_pad_cell)

    for x, y, cell in zip(cx, cy, cells):
        i = np.round((x - BORDER_SIZE - (size / 18)) / (size / 9)).astype(int)
        j = np.round((y - BORDER_SIZE - (size / 18)) / (size / 9)).astype(int)
        img_cell = img_table[x - h // 2 : x + h // 2, y - h // 2 : y + h // 2]
        
        if np.mean(img_cell * null_border_cell_20) > 10.:
            contours, hierarchy = cv2.findContours(np.uint8(img_cell * null_border_cell_10), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            areas = [cv2.contourArea(cnt) for cnt in contours]
            digit_idx = np.argsort(areas)[-1]
            y_min ,x_min, ww, hh = cv2.boundingRect(contours[digit_idx])
            step = 2
            img_cell = img_table[x - h // 2 + x_min - step : x - h // 2 + x_min + hh + step,
                                 y - h // 2 + y_min - step : y - h // 2 + y_min + ww + step]
            img_cell = np.pad(img_cell, pad_width=15)
        else:
            img_cell = img_cell * null_border_cell_20
        img_cell = cv2.resize(img_cell, (MNIST_CELL_SIZE, MNIST_CELL_SIZE), interpolation=cv2.INTER_AREA)
        digit_table[i][j] = img_cell * null_border
    return digit_table

        
def search_digits(img_table, digit_model, return_digit_images=False):
    size = img_table.shape[0] - 2 * BORDER_SIZE
    # find big contours and filter them to leave sudoku only
    contours, hierarchy = cv2.findContours(img_table, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    table_idx = np.argsort(areas)[-1]
    cells = np.array(contours)[np.argwhere(hierarchy[0].T[3] == table_idx).reshape(-1)]
    areas_cells = [cv2.contourArea(cnt) for cnt in cells]
    cells = cells[np.argwhere((areas_cells > np.median(areas_cells) * 0.9) & \
                                  (areas_cells < np.median(areas_cells) * 1.1)).reshape(-1)]
    if len(cells) == 81:
        digit_table = search_digits_good_table(img_table, cells)
    else:
        digit_table = search_digits_bad_table(img_table)
    
    for i in range(9):
        for j in range(9):
            if digit_table[i][j].max() - digit_table[i][j].min() < 255.:
                digit_table[i][j] = np.zeros((MNIST_CELL_SIZE, MNIST_CELL_SIZE))
            else:
                digit_table[i][j] = (digit_table[i][j] - np.median(digit_table[i][j])).clip(min=0)
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]
    )
    digit_images = {}
    for i in range(9):
        for j in range(9):
            img_cell = digit_table[i][j]
            if np.mean(img_cell) < 10.:
                digit_table[i][j] = -1
            else:
                img_cell = transform(img_cell / img_cell.max())[None, :]
                output = digit_model(img_cell)
                
                preds, digit = torch.max(output.data, 0)
                preds_idx = np.argsort(preds)
                if preds_idx[-1] == 5:
                    if preds[6] / preds[5] > 0.5:
                        preds_idx[-1] = 6
                if preds_idx[-1] == 7:
                    if preds[1] / preds[7] > 0.5:
                        preds_idx[-1] = 1
                digit_images[int(preds_idx[-1])] = digit_table[i][j]
                digit_table[i][j] = preds_idx[-1]

    digit_table = np.array(digit_table, dtype=np.int16)
    digit_table = np.where(digit_table == 0, 8, digit_table)
    if return_digit_images:
        return digit_table, digit_images
    return digit_table


def print_solution_to_table(normal_table, digit_table):
    sudoku_string = ''
    for row in digit_table:
        sudoku_row = ' '.join(map(str, row))
        sudoku_string += '\n' + sudoku_row.replace('-1', '?')

    # throws exception if unsolvable in case when the given table is broken
    solved_table = next(solve(sudoku_string))
    
    size = normal_table.shape[0] - 2 * BORDER_SIZE
    h = size // 10
    step = size // 100
    for i in range(9):
        for j in range(9):
            if digit_table[i][j] == -1:
                solution = DIGITS[solved_table[i][j]]
                solution = cv2.resize(solution, (h - 2 * step, h - 2 * step), interpolation=cv2.INTER_AREA)
                x = step * (i + 1) + h * i + BORDER_SIZE + step // 2
                y = step * (j + 1) + h * j + BORDER_SIZE + step // 2
                normal_table[x : x + h - 2 * step, y : y + h - 2 * step] = solution
    return normal_table
        

def draw_solution(img, normal_tables, initial_points, mask):
    normal_points = np.float32([[0, 0],
                                [0, NORMAL_TABLE_SIZE],
                                [NORMAL_TABLE_SIZE, 0], 
                                [NORMAL_TABLE_SIZE, NORMAL_TABLE_SIZE]])
    for k in range(len(normal_points)):
        normal_points[k] = [normal_points[k][0] + BORDER_SIZE, normal_points[k][1] + BORDER_SIZE]
    result = img.copy()
    for k in range(len(normal_tables)):
        transform = cv2.getPerspectiveTransform(normal_points, np.float32(initial_points[k]))
        initial_table = cv2.warpPerspective(normal_tables[k], transform, img.shape[::-1])
        result = (result - (initial_table - initial_table.min()) * mask).clip(min=0)
    # make black pixels a bit brighter
    result[result < 10] = np.random.randint(30, 40, size=result.shape)[result < 10]
    result = np.uint8(result)
    return result

    
def draw(img, filepath):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask, normal_tables, initial_points = search_tables(img)
    
    digit_model = MnistClassifier()
    digit_model.load_state_dict(torch.load(DIGIT_CLASSIFIER_PATH))
    digit_model.double()
    digit_model.eval()
    
    digits = []
    for k, normal_table in enumerate(normal_tables):
        digit_table = search_digits(normal_table, digit_model)
        try:
            normal_table = print_solution_to_table(normal_table, digit_table)
        except:
            print('Sudoku is unsolvable: unable to detect digits correctly.', file=sys.stderr)
            return
        digits.append(digit_table)
        normal_tables[k] = normal_table
        
    result = draw_solution(img, normal_tables, initial_points, mask)
    # plt.figure(dpi=150)
    # plt.imshow(img, cmap='gray');
    # plt.figure(dpi=150)
    # plt.imshow(result, cmap='gray');
    cv2.imwrite(filepath, result)
    return result
