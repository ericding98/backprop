import numpy as np

class Report(object):

  def __init__(
    self,
    pred,
    truth
  ):
    self.pred = pred
    self.truth = truth

  def accuracy(self):
    total = len(self.pred)
    correct = sum([
      self.pred[i] == self.truth[i]
      for i in range(total)
    ])
    return(correct/total)

  def truePositive(self):
    return({
      classKey: sum([
        self.pred[i] == self.truth[i]
        for i in range(len(self.pred))
        if self.pred[i] == classKey
      ])
      for classKey in set(self.truth)
    })

  def precision(self):
    tp = self.truePositive()
    return({
      classKey: tp[classKey] / (tp[classKey] + np.count_nonzero(self.pred == classKey) if np.count_nonzero(self.pred == classKey) != 0 else -1)
      for classKey in tp
    })

  def recall(self):
    tp = self.truePositive()
    return({
      classKey: tp[classKey] / (tp[classKey] + np.count_nonzero(self.truth == classKey))
      for classKey in tp
    })

  def f1(self):
    precision = self.precision()
    recall = self.recall()
    return({
      classKey: precision[classKey] * recall[classKey] / ((precision[classKey] + recall[classKey]) if (precision[classKey] + recall[classKey]) != 0 else -1)
      for classKey in precision
    })

  def confusion(self):
    matrix = []
    tp = self.truePositive()
    for row in tp:
      matrixRow = []
      for column in tp:
        if row == column:
          matrixRow.append(tp[column])
        else:
          preds = np.where(self.pred == row)[0]
          matrixRow.append(sum([
            self.truth[i] != self.pred[i]
            for i in preds
          ]))
      matrix.append(matrixRow)
    return(np.array(matrix))

  def multiClass(self):
    return(np.count_nonzero(self))
