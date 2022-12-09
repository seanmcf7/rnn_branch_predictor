import numpy as np

class SRNN():
    def __init__(self, pht_size, pht_dtype=np.int8, pht_update_weight=1):
        self.ghr_size = 32
        self.pht_size = np.uint16(pht_size)
        self.pht_index_mask = self.pht_size - np.uint16(1)
        self.pht_update_weight = pht_update_weight
        self.last_prediction = 0

        valid_dtypes = [np.int8, np.int16, np.int32, np.int64]
        if pht_dtype not in valid_dtypes:
            raise ValueError(f'Invalid pht_dtype: {pht_dtype}')

        # 32 bit global history register
        self.ghr = np.ones(self.ghr_size, dtype=np.int8)

        # Pattern history table
        self.pht_w = np.random.randint(low=-16, high=16, size=(self.pht_size,32), dtype=pht_dtype)
        self.pht_u = np.random.randint(low=-1, high=2, size=(self.pht_size,31), dtype=pht_dtype)

        if pht_dtype is np.int8:
            self.pht_min = -128
            self.pht_max = 127
        elif pht_dtype is np.int16:
            self.pht_min = -32768
            self.pht_max = 32767
        elif pht_dtype is np.int32:
            self.pht_min = -2147483648
            self.pht_max = 2147483647
        elif pht_dtype is np.int64:
            self.pht_min = -9223372036854775808
            self.pht_max = 9223372036854775807


    def predict(self, pc):
        s_val = np.zeros(32, dtype=np.int64)
        w_val = self.pht_w[(int(pc) >> 2) & self.pht_index_mask]
        u_val = self.pht_u[(int(pc) >> 2) & self.pht_index_mask]
        u_index = 0
        u_increment = 2

        # Fill initial weights 

        for i in range(32):
            s_val[i] = self.ghr[i] * w_val[u_index]
                

        # Fill layers
        s_count = 16
        while s_count > 0:
            for i in range(s_count):
                s_val[i] = s_val[i<<1] * u_val[u_index] + s_val[(i<<1) + 1]
                u_index += u_increment
            u_index = u_increment - 1
            u_increment = u_increment << 1
            s_count = s_count >> 1

        # Final prediction is in s_val[0]
        self.last_prediction = s_val[0]
        prediction = 1
        if s_val[0] < 0:
            prediction = 0

        return prediction


    def update_ght(self, actual):
        for i in range(0, self.ghr_size-1):
            self.ghr[i] = self.ghr[i + 1]
        self.ghr[self.ghr_size-1] = actual


    def update_pht(self, pc, predicted, actual):
        if predicted == actual:
            return
        w_val = self.pht_w[(int(pc) >> 2) & self.pht_index_mask]
        u_val = self.pht_u[(int(pc) >> 2) & self.pht_index_mask]
        update_thresh = 10
        u_index = 0
        u_increment = 2

        if actual == 0:
            actual = -1

        if abs(self.last_prediction) < update_thresh or predicted != actual:
            # Increment weight if it had the correct prediction
            for i in range(32):
                if self.ghr[i] == actual and w_val[i] < (self.pht_max - self.pht_update_weight):
                    w_val[i] += self.pht_update_weight
                elif w_val[i] > (self.pht_min + self.pht_update_weight):
                    w_val[i] -= self.pht_update_weight

            for i in range(31):
                if self.ghr[i] == actual and u_val[u_index] < self.pht_max:
                    u_val[u_index] += 1
                else:
                    u_val[u_index] = 1
                u_index += u_increment
                if u_index >= 31:
                    u_index = u_increment - 1
                    u_increment = u_increment << 1       
