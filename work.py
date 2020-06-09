import UserCF
import ItemCF
rating_file = './ml-1m/ratings.dat'
userCF = UserCF.UserBasedCF()
userCF.get_dataset(rating_file)
userCF.calc_user_sim()
userCF.evaluate()

def MixCF():
    rating_file = './ml-1m/ratings.dat'
    userCF = UserCF.UserBasedCF()
    userCF.get_dataset(rating_file)
    userCF.calc_user_sim()
    itemCF = ItemCF.ItemBasedCF()
    itemCF.get_dataset(rating_file)
    itemCF.calc_movie_sim()
    itemCF.evaluate()