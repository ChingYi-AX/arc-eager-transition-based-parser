import numpy as np

"""FeatureMapping: create feature template"""


class FeatureMapping:
    def __init__(self):
        self.frozen = False  # freeze the mapping function after finish training
        self.id = 1
        self.feature_map = dict()  # Dict['str':int]

    def extract_feature(self, feature):
        if self.frozen is True:
            if feature not in self.feature_map:
                return 0
            else:
                return self.feature_map[feature]
        elif self.frozen is False:
            if feature not in self.feature_map:
                self.feature_map[feature] = self.id
                self.id += 1
            return self.feature_map[feature]

    def feature_template(self, state, sentence):
        features = np.empty(0, dtype=int)

        form = sentence.form
        pos = sentence.pos
        lemma = sentence.lemma
        head = state.arc

        stack = state.stack
        buffer = state.buffer
        ld = state.left_most
        rd = state.right_most

        if stack and buffer:
            stack_top = stack[-1]
            buffer_bottom = buffer[0]
            ''' Both stack and buffer '''
            features = np.append(features, self.extract_feature(
                "s[0]-form-pos+b[0]-form-pos=" + form[stack_top] + pos[stack_top] + form[buffer_bottom] + pos[buffer_bottom]))
            features = np.append(features, self.extract_feature("s[0]-form-pos+b[0]-form=" + form[stack_top] + pos[stack_top] + form[buffer_bottom]))
            features = np.append(features, self.extract_feature("s[0]-form+b[0]-form-pos=" + form[stack_top] + form[stack_top] + pos[buffer_bottom]))
            features = np.append(features, self.extract_feature("s[0]-form-pos+b[0]-pos=" + form[stack_top] + pos[stack_top] + pos[buffer_bottom]))
            features = np.append(features, self.extract_feature("s[0]-pos+b[0]-form-pos=" + pos[stack_top] + form[buffer_bottom] + pos[buffer_bottom]))
            features = np.append(features, self.extract_feature("s[0]-form+b[0]-form=" + form[stack_top] + form[buffer_bottom]))
            features = np.append(features, self.extract_feature("s[0]-pos+b[0]-pos=" + pos[stack_top] + pos[buffer_bottom]))
            features = np.append(features, self.extract_feature("s[0]-lemma+b[0]-lemma=" + lemma[stack_top] + lemma[buffer_bottom]))

            if head[stack_top] >= 0:
                features = np.append(features, self.extract_feature(
                    "s[0]-pos+b[0]-pos+hd-s[0]-pos=" + pos[stack_top] + pos[stack_top] + pos[head[stack_top]]))
            if ld[buffer_bottom] >= 0:
                features = np.append(features, self.extract_feature("s[0]-pos+b[0]-pos+ld-s[0]-pos=" + pos[stack_top] + pos[buffer_bottom] + pos[ld[stack_top]]))
                features = np.append(features, self.extract_feature("s[0]-pos+b[0]-pos+ld-b[0]-pos=" + pos[stack_top] + pos[buffer_bottom] + pos[ld[buffer_bottom]]))
            if rd[buffer_bottom] >= 0:
                features = np.append(features, self.extract_feature("s[0]-pos+b[0]-pos+rd-s[0]-pos=" + pos[stack_top] + pos[buffer_bottom] + pos[rd[stack_top]]))
                features = np.append(features, self.extract_feature("s[0]-pos+b[0]-pos+rd-b[0]-pos=" + pos[stack_top] + pos[buffer_bottom] + pos[rd[buffer_bottom]]))

        if stack:
            stack_top = stack[-1]
            features = np.append(features, self.extract_feature("s[0]-form=" + form[stack_top]))
            features = np.append(features, self.extract_feature("s[0]-pos=" + pos[stack_top]))
            features = np.append(features, self.extract_feature("s[0]-lemma=" + lemma[stack_top]))
            features = np.append(features, self.extract_feature("s[0]-form-pos=" + form[stack_top] + pos[stack_top]))
            features = np.append(features, self.extract_feature("s[0]-lemma-pos=" + lemma[stack_top] + pos[stack_top]))

            if len(stack) > 1:
                second_top_s = stack[-2]
                features = np.append(features, self.extract_feature("s[1]-form=" + form[second_top_s]))
                features = np.append(features, self.extract_feature("s[1]-pos=" + pos[second_top_s]))
                features = np.append(features, self.extract_feature("s[1]-form-pos=" + form[second_top_s] + pos[second_top_s]))

            if head[stack_top] >= 0:
                features = np.append(features, self.extract_feature("head-s[0]-form=" + form[head[stack_top]]))
                features = np.append(features, self.extract_feature("head-s[0]-pos=" + pos[head[stack_top]]))
            if ld[stack_top] >= 0:
                features = np.append(features, self.extract_feature("ld-s[0]-form=" + form[ld[stack_top]]))
                features = np.append(features, self.extract_feature("ld-s[0]-pos=" + pos[ld[stack_top]]))
            if rd[stack_top] >= 0:
                features = np.append(features, self.extract_feature("rd-s[0]-form=" + form[rd[stack_top]]))
                features = np.append(features, self.extract_feature("rd-s[0]-pos=" + pos[rd[stack_top]]))

        if buffer:
            buffer_bottom = buffer[0]
            features = np.append(features, self.extract_feature("b[0]-form=" + form[buffer_bottom]))
            features = np.append(features, self.extract_feature("b[0]-pos=" + pos[buffer_bottom]))
            features = np.append(features, self.extract_feature("b[0]-lemma=" + lemma[buffer_bottom]))
            features = np.append(features, self.extract_feature("b[0]-form-pos=" + form[buffer_bottom] + pos[buffer_bottom]))
            features = np.append(features, self.extract_feature("b[0]-lemma-pos=" + lemma[buffer_bottom] + pos[buffer_bottom]))

            if head[buffer_bottom] >= 0:
                features = np.append(features, self.extract_feature("head-b[0]-form=" + form[head[buffer_bottom]]))
                features = np.append(features, self.extract_feature("head-b[0]-pos=" + pos[head[buffer_bottom]]))
            if ld[buffer_bottom] >= 0:
                features = np.append(features, self.extract_feature("ld-b[0]-form=" + form[ld[buffer_bottom]]))
                features = np.append(features, self.extract_feature("ld-b[0]-pos=" + pos[ld[buffer_bottom]]))
            if rd[buffer_bottom] >= 0:
                features = np.append(features, self.extract_feature("rd-b[0]-form=" + form[rd[buffer_bottom]]))
                features = np.append(features, self.extract_feature("rd-b[0]-pos=" + pos[rd[buffer_bottom]]))

            if len(buffer) > 1:
                buffer_bottom_2 = buffer[1]
                features = np.append(features, self.extract_feature("b[1]-form=" + form[buffer_bottom_2]))
                features = np.append(features, self.extract_feature("b[1]-pos=" + pos[buffer_bottom_2]))
                features = np.append(features, self.extract_feature("b[1]-form-pos=" + form[buffer_bottom_2] + pos[buffer_bottom_2]))
                features = np.append(features, self.extract_feature("b[0]-form+b[1]-form=" + form[buffer_bottom_2] + form[buffer_bottom_2]))
                features = np.append(features, self.extract_feature("b[0]-pos+b1-pos=" + pos[buffer_bottom_2] + pos[buffer_bottom_2]))
                if stack:
                    features = np.append(features, self.extract_feature(
                        "b[0]-pos+b[1]-pos+s[0]-pos=" + pos[buffer_bottom_2] + pos[buffer_bottom_2] + pos[stack[-1]]))

            if len(buffer) > 2:
                buffer_bottom_3 = buffer[2]
                features = np.append(features, self.extract_feature("b[2]-pos=" + pos[buffer_bottom_3]))
                features = np.append(features, self.extract_feature("b[2]-form=" + form[buffer_bottom_3]))
                features = np.append(features, self.extract_feature("b[2]-form-pos=" + form[buffer_bottom_3] + pos[buffer_bottom_3]))
                features = np.append(features, self.extract_feature("b[0]-pos+b[1]-pos+b[2]-pos=" + pos[buffer_bottom_3] + pos[buffer_bottom_3] + pos[buffer_bottom_3]))

            if len(buffer) > 3:
                buffer_bottom_4 = buffer[3]
                features = np.append(features, self.extract_feature("b[3]-pos=" + pos[buffer_bottom_4]))

        return features

