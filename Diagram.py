import random
import numpy as np

class Diagram:
    
    def __init__(self):
        self.isDangerous = False
        self.third = None
        self.image = self.generateImage()


    def generateImage(self):
        p = random.random()
        if p > 0.5:
            image = self.generateRowFirst()
        else:
            image = self.generateColFirst()
            
        return image
    
    def generateDangerousImage(self):
        while not self.isDangerous:  
            p = random.random()
            if p > 0.5:
                self.image = self.generateRowFirst()
            else:
                self.image = self.generateColFirst()
    
    def generateRowFirst(self):
        colors = ['R', 'B', 'Y', 'G']
        image = [['E']*20]*20
        image = np.array(image)
        
        #First color
        row = random.randint(0,19)
        color = random.choice(colors)
        image[row] = color
        colors.remove(color)
        self.checkIfDangerous(color, colors)
        
        #Second color
        col = random.randint(0,19)
        color = random.choice(colors)
        for i in range(20):
            image[:,col] = color
        colors.remove(color)
        self.checkIfDangerous(color, colors)
        
        #Third color
        row2 = row
        while row2 == row:
            row2 = random.randint(0,19)
        color = random.choice(colors)
        image[row2] = color
        colors.remove(color)
        self.checkIfDangerous(color, colors)
        self.third = color
        
        #Last color
        color = colors.pop(0)
        col2 = col
        while col2 == col:
            col2 = random.randint(0,19)
        for i in range(20):
            image[:, col2] = color
            
        return image        
    
    def generateColFirst(self):
        colors = ['R', 'B', 'Y', 'G']
        image = [['E']*20]*20
        image = np.array(image)
        
        #Second color
        col = random.randint(0,19)
        color = random.choice(colors)
        for i in range(20):
            image[:,col] = color
        colors.remove(color)
        self.checkIfDangerous(color, colors)
        
        #First color
        row = random.randint(0,19)
        color = random.choice(colors)
        image[row] = color
        colors.remove(color)
        self.checkIfDangerous(color, colors)
        
        #Last color
        color = colors.pop(0)
        col2 = col
        while col2 == col:
            col2 = random.randint(0,19)
        for i in range(20):
            image[:, col2] = color
        self.third = color
    
        #Third color
        row2 = row
        while row2 == row:
            row2 = random.randint(0,19)
        color = random.choice(colors)
        image[row2] = color
        colors.remove(color)
        self.checkIfDangerous(color, colors)
        
        
        return image
    
    def printImage(self):
        image = self.image
        
        for i in range(20):
            for j in range(20):
                print(image[i][j], end=" ")
            print('\n', end="")
    
    def checkIfDangerous(self,color, remainingColors):
        if color == 'R' and 'Y' in remainingColors:
            self.isDangerous = True
            
        
