import math
import pyautogui, time
from screeninfo import get_monitors
import numpy as np
import matplotlib.pyplot as plt


def regression_find_W(data_x, data_y):
    result = np.round(np.dot(np.dot(np.linalg.inv(np.dot(data_x.transpose(), data_x)), data_x.transpose()), data_y), 4)
    return result


def regression_cost_function(orginalData):
    costFunction = 0
    for point in orginalData:
        error = (point[1] - (W1*point[0] + W0))**2
        costFunction += error
    return costFunction


def mean_square_error(costFunction, dataLengh):
    return costFunction/dataLengh


def root_mean_square_error(MSE):
    return math.sqrt(MSE)


# Detect Screen Size
print('Screen:')
for m in get_monitors():
    print('Name:' + str(m.name))
    screenHeight = int(m.height)
    print('H:' + str(m.height))
    screenWidth = int(m.width)
    print('W:' + str(m.width))
    break
    # we don't need to other screens

inputForPosition = int(input("\nEnter a time limit for get Mouse Position Data(s): "))

print("\nGet Ready !...\n");time.sleep(1);print("-3-");time.sleep(1);print("-2-");time.sleep(1);print("-1-\n")

orginalData = list()
x = list()
y = list()
plt_x = list()
plt_y = list()

# Get Mouse Position:
for position in range(0,inputForPosition*4):
    time.sleep(0.25)
    orginalData.append([pyautogui.position().x,(screenHeight - pyautogui.position().y)])
    x.append([1, pyautogui.position().x])
    y.append([screenHeight - pyautogui.position().y])
    plt_x.append(pyautogui.position().x)
    plt_y.append(screenHeight - pyautogui.position().y)

data_x = np.array(x)
data_y = np.array(y)

# # Sample:
# orginalData = np.array([[0, 0], [2, 2], [1, 1], [0, 1]])
# data_x = np.array([[1, 0], [1, 2], [1, 1], [1, 0]])
# data_y = np.array([[0], [2], [1], [1]])
# plt_x = [0, 2, 1, 0]
# plt_y = [0, 2, 1, 1]

print('Orginal Data = \n' + str(np.array(orginalData)))
input('\nDone. Press Enter to continue ... ')

print('\nx =\n' + str(data_x))
print('\ny = \n' + str(data_y))

# Calculate W:
W = regression_find_W(data_x, data_y)
W0 = W[0][0]
W1 = W[1][0]
print(f'\nW =\n[[{W0}]\n [{W1}]]')
print(f'\ny = {W0} + {W1}x')

# Calculate Cost Function:
costFunction = regression_cost_function(orginalData)
print(f'\nCost Function = {costFunction}')

MSE = mean_square_error(costFunction, len(orginalData))
print(f'MSE = {MSE}')

RMSE = root_mean_square_error(MSE)
print(f'RMSE = {RMSE}')

# Draw Diagram of regresion:
xRange = np.arange(0, screenWidth, 0.01)
plt.plot(xRange, W1*xRange + W0, label='regression')

plt.scatter(plt_x, plt_y, label= "Samples", color= "green", marker= "*")

plt.xlabel('x')
plt.ylabel('y')
plt.title(f'y = {W0} + {W1}x // RMSE = {round(RMSE, 4)}')
plt.legend()

plt.show()
