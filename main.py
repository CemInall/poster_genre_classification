from data import get_data
from ml_models import do_machine_learning
from grid_searches import do_all_grid_searches
from CNN import do_deep_learning
movielabels, images = get_data()
#do_all_grid_searches(movielabels, images)
do_deep_learning(movielabels,images)
#do_machine_learning(movielabels,images)
