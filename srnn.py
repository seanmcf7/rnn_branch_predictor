import numpy as np

class SRNN():
    def __init__(self):
        self.ghr_size = 32
        self.last_prediction = 0

        # 32 bit global history register
        self.ghr = np.ones(self.ghr_size, dtype=np.int8)

        # Pattern history table
        self.pht_w = np.random.randint(low=-16, high=16, size=(256,32), dtype=np.int8)
        self.pht_u = np.random.randint(low=-1, high=2, size=(256,31), dtype=np.int8)


    def predict(self, pc):
        s_val = np.zeros(32, dtype=np.int64)
        w_val = self.pht_w[pc & np.uint64(0x1F)]
        u_val = self.pht_u[pc & np.uint64(0x1F)]
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
        w_val = self.pht_w[pc & np.uint64(0x1F)]
        u_val = self.pht_u[pc & np.uint64(0x1F)]
        update_thresh = 10
        u_index = 0
        u_increment = 2

        if actual == 0:
            actual = -1

        if abs(self.last_prediction) < update_thresh or predicted != actual:
            # Increment weight if it had the correct prediction
            for i in range(32):
                if self.ghr[i] == actual:
                    w_val[i] += 1
                else:
                    w_val[i] -= 1

            for i in range(31):
                if self.ghr[i] == actual:
                    u_val[u_index] += 1
                else:
                    u_val[u_index] = 1
                u_index += u_increment
                if u_index >= 31:
                    u_index = u_increment - 1
                    u_increment = u_increment << 1       
        
        self.update_ght(actual) 
