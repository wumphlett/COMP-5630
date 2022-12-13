#!/usr/bin/python
from pathlib import Path

from PIL import Image, ImageStat
import numpy
import matplotlib.pyplot as plt


def converged(centroids, old_centroids):
    if len(old_centroids) == 0:
        return False

    if len(centroids) <= 5:
        a = 1
    elif len(centroids) <= 10:
        a = 2
    else:
        a = 4

    for i in range(0, len(centroids)):
        cent = centroids[i]
        old_cent = old_centroids[i]

        if ((int(old_cent[0]) - a) <= cent[0] <= (int(old_cent[0]) + a)) and ((int(old_cent[1]) - a) <= cent[1] <= (int(old_cent[1]) + a)) and ((int(old_cent[2]) - a) <= cent[2] <= (int(old_cent[2]) + a)):
            continue
        else:
            return False

    return True


def getMin(pixel, centroids):
    minDist = 9999
    minIndex = 0

    for i in range(0, len(centroids)):
        d = numpy.sqrt(int((centroids[i][0] - pixel[0]))**2 + int((centroids[i][1] - pixel[1]))**2 + int((centroids[i][2] - pixel[2]))**2)
        if d < minDist:
            minDist = d
            minIndex = i

    return minIndex, minDist


def assignPixels(centroids):
    clusters = {}

    for x in range(img_width):
        for y in range(img_height):
            pixel = px[x, y]
            found_min, _ = getMin(pixel, centroids)
            if found_min in clusters:
                clusters[found_min].append(pixel)
            else:
                clusters[found_min] = [pixel]

    return clusters


def adjustCentroids(clusters):
    new_centroids = []

    for key in clusters:
        mean = numpy.mean(clusters[key], axis=0)
        new_centroids.append((int(mean[0]), int(mean[1]), int(mean[2])))

    return new_centroids


def initializeKmeans(someK):
    centroids = []

    for k in range(someK):
        centroid = px[numpy.random.randint(0, img_width), numpy.random.randint(0, img_height)]
        centroids.append(centroid)

    print("Centroids Initialized")
    print("===========================================")

    return centroids


def iterateKmeans(centroids):
    old_centroids = []
    print("Starting Assignments")
    print("===========================================")

    while not converged(centroids, old_centroids):
        old_centroids = centroids
        clusters = assignPixels(centroids)
        centroids = adjustCentroids(clusters)

    print("===========================================")
    print("Convergence Reached!")
    return centroids


def drawWindow(result):
    img = Image.new('RGB', (img_width, img_height), "white")
    p = img.load()

    for x in range(img.size[0]):
        for y in range(img.size[1]):
            RGB_value = result[getMin(px[x, y], result)[0]]
            p[x, y] = RGB_value

    img.show()
    return img


if __name__ == '__main__':
    if_assignment_3 = str(input("Would you like to run question 3? y/n: "))

    if if_assignment_3:
        cwd = Path.cwd()

        # 3a
        im = Image.open("img/test05.jpg")
        img_width, img_height = im.size
        px = im.load()

        question_3a = cwd / "3a"
        question_3a.mkdir(exist_ok=True)

        for i in range(10):
            initial_centroid = initializeKmeans(2)
            result = iterateKmeans(initial_centroid)
            result_img = drawWindow(result)
            result_img.save(question_3a / f"{i}.jpg")

        # 3b
        im = Image.open("img/test05.jpg")
        img_width, img_height = im.size
        px = im.load()

        question_3b = cwd / "3b"
        question_3b.mkdir(exist_ok=True)

        for i in range(10):
            initial_centroid = initializeKmeans(10)
            result = iterateKmeans(initial_centroid)
            result_img = drawWindow(result)
            result_img.save(question_3b / f"{i}.jpg")

        # 3c
        im = Image.open("img/test08.jpg")
        img_width, img_height = im.size
        px = im.load()

        question_3c = cwd / "3c"
        question_3c.mkdir(exist_ok=True)

        for i in range(10):
            for k in range(2, 6):
                initial_centroid = initializeKmeans(k)
                result = iterateKmeans(initial_centroid)
                result_img = drawWindow(result)
                result_img.save(question_3c / f"{k}-{i}.jpg")

    else:
        num_input = str(input("Enter image number: "))
        if_assignment_2 = str(input("Would you like to run question 2? y/n: "))

        img = "img/test" + num_input.zfill(2) + ".jpg"
        im = Image.open(img)
        img_width, img_height = im.size
        px = im.load()

        if if_assignment_2 != "y":
            k_input = int(input("Enter K value: "))
            initial_centroid=initializeKmeans(k_input)
            result = iterateKmeans(initial_centroid)
            drawWindow(result)
        else:
            plotX = []
            plotY = []
            objective = 0

            for i in range(1, 21):
                plotX.append(i)
                initial_centroid = initializeKmeans(i)
                centroids = iterateKmeans(initial_centroid)
                for x in range(img_width):
                    for y in range(img_height):
                        objective += numpy.square(getMin(px[x, y], centroids)[1])
                plotY.append(objective)

            fig, ax = plt.subplots(1, 1)
            ax.scatter(plotX,
                       plotY,
                       color="red",
                       linewidths=0.5,
                       label="Points")
            ax.set_xlabel("K Value")
            ax.set_ylabel("K-Means Objective")
            plt.savefig('k-means_plot.png')
            plt.close()
