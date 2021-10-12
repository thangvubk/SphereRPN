from abc import ABCMeta, abstractmethod


class BaseAssigner(metaclass=ABCMeta):

    @abstractmethod
    def assign(self, boxes, gt_boxes, gt_boxes_ignore=None, gt_labels=None):
        pass
