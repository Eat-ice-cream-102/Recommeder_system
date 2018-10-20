# coding: utf-8 -*-
import math
import pandas as pd


class UserCf:

    def __init__(self):
        self.file_path = 'data/ratings.csv'
        self._init_frame()

    def _init_frame(self):
        self.frame = pd.read_csv(self.file_path)

    @staticmethod
    def _cosine_sim(target_products, products):
        '''
        simple method for calculate cosine distance.
        e.g: x = [1 0 1 1 0], y = [0 1 1 0 1]
             cosine = (x1*y1+x2*y2+...) / [sqrt(x1^2+x2^2+...)+sqrt(y1^2+y2^2+...)]
             that means union_len(products1, products2) / sqrt(len(products1)*len(products2))
        '''
        union_len = len(set(target_products) & set(products))
        if union_len == 0: return 0.0
        product = len(target_products) * len(products)
        cosine = union_len / math.sqrt(product)
        return cosine

    def _get_top_n_users(self, target_user_id, top_n):
        '''
        calculate similarity between all users and return Top N similar users.
        '''
        target_products = self.frame[self.frame['UserID'] == target_user_id]['productID']
        other_users_id = [i for i in set(self.frame['UserID']) if i != target_user_id]
        other_products = [self.frame[self.frame['UserID'] == i]['productID'] for i in other_users_id]

        sim_list = [self._cosine_sim(target_products, products) for products in other_products]
        sim_list = sorted(zip(other_users_id, sim_list), key=lambda x: x[1], reverse=True)
        return sim_list[:top_n]

    def _get_candidates_items(self, target_user_id):
        """
        Find all products in source data and target_user did not meet before.
        """
        target_user_products = set(self.frame[self.frame['UserID'] == target_user_id]['productID'])
        other_user_products = set(self.frame[self.frame['UserID'] != target_user_id]['productID'])
        candidates_products = list(target_user_products ^ other_user_products)
        return candidates_products

    def _get_top_n_items(self, top_n_users, candidates_products, top_n):
        """
        calculate interest of candidates products and return top n products.
        e.g. interest = sum(sim * normalize_rating)
        """
        top_n_user_data = [self.frame[self.frame['UserID'] == k] for k, _ in top_n_users]
        interest_list = []
        for product_id in candidates_products:
            tmp = []
            for user_data in top_n_user_data:
                if product_id in user_data['productID'].values:
                    tmp.append(user_data[user_data['productID'] == product_id]['Rating'].values[0]/5)
                else:
                    tmp.append(0)
            interest = sum([top_n_users[i][1] * tmp[i] for i in range(len(top_n_users))])
            interest_list.append((product_id, interest))
        interest_list = sorted(interest_list, key=lambda x: x[1], reverse=True)
        return interest_list[:top_n]

    def calculate(self, target_user_id=1, top_n=10):
        """
        user-cf for products recommendation.
        """
        # most similar top n users
        top_n_users = self._get_top_n_users(target_user_id, top_n)
        # candidates products for recommendation
        candidates_products = self._get_candidates_items(target_user_id)
        # most interest top n products
        top_n_products = self._get_top_n_items(top_n_users, candidates_products, top_n)
        return top_n_products
