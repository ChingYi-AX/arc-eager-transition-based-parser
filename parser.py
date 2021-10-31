import numpy as np
from collections import deque


'''Parsing: check parsing result with oracle parsing during training'''


class Parsing:
    def __init__(self, state, sentence, featuremap):

        self.state = state
        self.sentence = sentence
        self.map = featuremap
        self.training_data = list()

    def parse(self, loaded_model):

        weight = loaded_model.weight
        while self.state.buffer:
            scores = np.asarray([0., 0., 0., 0.])  # initialize score
            feature_list = self.map.feature_template(self.state, self.sentence)

            for feature in feature_list:
                for ix in range(0, 4):
                    scores[ix] += weight[ix][feature]  # update score

            pred_transition = np.argsort(-scores) # sort score form high to low

            for trans in pred_transition:

                if trans == 1 and self.state.stack and self.can_left_arc():
                    self.left_arc()
                    break
                elif trans == 0 and self.state.stack:
                    self.right_arc()
                    break
                elif trans == 2 and self.state.stack and self.can_reduce():
                    self.reduce()
                    break
                elif trans == 3 and self.state.stack:
                    self.shift()
                    break

        # We have to add to left-neighbour if there is headless word, but do not consider root token
        for ix in range(1, len(self.state.arc)):
            if self.state.arc[ix] == -1:
                self.state.arc[ix] = ix-1
        return self.state.arc

    def oracle_parser(self):
        prepare_data = []  # use oracle parser to check the result and thus compute train data

        while self.state.buffer:  # if buffer is not empty
            feature_template = self.map.feature_template(self.state, self.sentence)
            feature_list = np.asarray(feature_template)

            if self.state.stack and self.should_right_arc():
                inst = Instance(0, feature_list)
                self.right_arc()  # do right arc

            elif self.state.stack and self.should_left_arc():
                inst = Instance(1, feature_list)
                self.left_arc()  # do left arc

            elif self.state.stack and self.should_reduce():
                inst = Instance(2, feature_list)
                self.reduce()  # do reduce
            else:
                inst = Instance(3, feature_list)
                self.shift()  # do shift

            prepare_data.append(inst)
        self.training_data = prepare_data

        return self.training_data  # later use for train the model

    '''should_xx(): oracle rules'''

    def should_left_arc(self):
        buff_bottom = self.state.buffer[0]
        stack_top = self.state.stack[-1]
        if (buff_bottom, stack_top) in self.sentence.gold_arc:
            return True
        return False

    def should_right_arc(self):
        buffer_bottom = self.state.buffer[0]
        stack_top = self.state.stack[-1]
        if (stack_top, buffer_bottom) in self.sentence.gold_arc:
            return True
        return False

    def should_reduce(self):
        stack_top = self.state.stack[-1]

        if self.has_head(stack_top) and self.has_all_children(stack_top):
            return True
        return False

    ''' can_xx(): pre-condition to determine whether a transition is valid (not necessarily)'''

    def can_reduce(self):
        stack_top = self.state.stack[-1]
        if self.state.arc[stack_top] != -1:
            return True
        return False

    def can_shift(self):
        if self.state.stack or len(self.state.buffer) >= 1 :
            return True
        return False

    def can_left_arc(self):
        stack_top = self.state.stack[-1]
        if self.state.arc[stack_top] == -1 and stack_top != 0:
            return True
        return False

    """has_all_children(): check before attach to a head"""

    def has_all_children(self, arc):
        for gold_arc in self.sentence.gold_arc:
            head = gold_arc[0]
            dependent = gold_arc[1]
            if head == arc:
                if self.state.arc[dependent] != arc:
                    return False
        return True

    '''has_head(): current arc '''

    def has_head(self, arc):
        number = 0
        if self.state.arc[arc] != -1:
            number += 1

        if number < 1:
            return False
        else:
            return True

    '''do_xx(): execute the transition after checking every condition'''

    def left_arc(self):
        remain_stack = self.state.stack.pop()
        self.state.arc[remain_stack] = self.state.buffer[0]

        left_most = min(self.state.get_transition(self.state.buffer[0]))
        self.state.left_most[self.state.buffer[0]] = left_most
        right_most = max(self.state.get_transition(self.state.buffer[0]))
        self.state.right_most[self.state.buffer[0]] = right_most

    def right_arc(self):
        stack_top = self.state.stack[-1]
        self.state.arc[self.state.buffer[0]] = stack_top
        left_most = min(self.state.get_transition(stack_top))
        self.state.left_most[stack_top] = left_most
        right_most = max(self.state.get_transition(stack_top))
        self.state.right_most[stack_top] = right_most

        self.state.stack.append(self.state.buffer[0])
        self.state.buffer.popleft()

    def shift(self):
        top_buffer = self.state.buffer.popleft()
        self.state.stack.append(top_buffer)

    def reduce(self):
        self.state.stack.pop()


'''Instance: take transition labels based on features'''


class Instance:
    def __init__(self, label, feature_vector):
        self.label = label  # correct label
        self.feature_vector = feature_vector  # Dict["int": int]: sparse representation


'''State: initialize initial stack, buffer, arc, left-dependency(left_most) and right-dependency(right_most)'''


class State:
    def __init__(self, stack, buffer):
        self.stack = deque(stack)
        self.buffer = deque(buffer)

        array_shape = len(self.buffer)+len(self.stack)
        self.arc = np.empty(array_shape, dtype=np.int32)
        self.left_most = np.empty(array_shape, dtype=np.int32)
        self.right_most = np.empty(array_shape, dtype=np.int32)

        self.arc.fill(-1)  # initial value: -1
        self.left_most.fill(-1)  # initial value: -1
        self.right_most.fill(-1)  # initial value: -1

    def get_transition(self, head):
        transition_list = []
        for ix in range(0, len(self.arc)):
            if self.arc[ix] == head:
                transition_list.append(ix)
                return transition_list

