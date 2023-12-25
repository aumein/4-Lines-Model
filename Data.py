import Diagram as d
import numpy as np

class Data: 
    def __init__(self, n):
        self.dataset, self.labels = self.generateDataset(n)    
        
    def convertDiagramToInput(self, diagram):
        arrayRepresentation = []
        
        result = diagram.isDangerous    #BOOL, to compare with netw result
        image = diagram.image
        
        for row in image:
            for i in row:
                chunk = self.colorToArr(i)
                arrayRepresentation.append(chunk)
                
        return np.array(arrayRepresentation).flatten()
                
                
    def colorToArr(self, color):
        match color:
            case 'R':
                return np.array([1,0,0,0])
            case 'Y':
                return np.array([0,1,0,0])
            case 'B':
                return np.array([0,0,1,0])
            case 'G':
                return np.array([0,0,0,1])
            case 'E':
                return np.array([0,0,0,0])
            
        return None
    
    def generateDataset(self, n):
        dataset = np.zeros(shape=(n, 1600))
        labels = np.zeros(shape=(n, 2))

        for i in range(n):
            diagram = d.Diagram()
            nextInput = self.convertDiagramToInput(diagram)
            dataset[i] = nextInput
            match diagram.isDangerous:
                case True:
                    labels[i] = [1,0]
                case False:
                    labels[i] = [0,1]
        return dataset, labels
    
    def generateDataset2(self, n):
        dataset = np.zeros(shape=(n, 1600))
        labels = np.zeros(shape=(n, 4))

        for i in range(n):
            diagram = d.Diagram()
            diagram.generateDangerousImage()
            nextInput = self.convertDiagramToInput(diagram)
            dataset[i] = nextInput
            match diagram.third:
                case 'R':
                    labels[i] = [1,0,0,0]
                case 'Y':
                    labels[i] = [0,1,0,0]
                case 'B':
                    labels[i] = [0,0,1,0]
                case 'G':
                    labels[i] = [0,0,0,1]
                    
        return dataset, labels
            
def main():
    asdf = Data(1)
    dataset, labels = asdf.generateDataset2(1)
    print(labels)
    
if __name__ == "__main__":
    main()
    
