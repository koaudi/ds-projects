import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 

from settings import MAIN_DIRECTORY
os.chdir(MAIN_DIRECTORY)
os.getcwd()

ke_skus = pd.read_csv("csv/ke/ke_product_skus.csv",skipinitialspace=True)
ke_skus.rename(columns = { 'Description' : 'description', 'Name' : 'product_name'}, inplace = True)
ds = ke_skus[['product_id','product_name','description']]

#
# rw_skus = pd.read_csv("csv/rw/rw_product_skus.csv",skipinitialspace=True)
# rw_skus.rename(columns = { 'Description' : 'description', 'Name' : 'product_name'}, inplace = True)
# ds = rw_skus[['product_id','product_name','description']]

#CLEAN PRODUCT DESCRIPTIONS
ds.description = ds.description.str.replace('''<span style="font-weight: 400">''',"")
ds.description = ds.description.str.replace('''<span style="font-weight: 400;">''',"")
ds.description = ds.description.str.replace('''<strong>''',"")
ds.description = ds.description.str.replace('''</span>''',"")
ds.description = ds.description.str.strip('/')

ds.dropna(subset=['description'], inplace = True)
ds = ds.reset_index()


#TERM FREQUENCY The TF*IDF algorithm is used to weigh a keyword in any document and assign the importance to that keyword based on the number of times it appears in the document. Put simply, the higher the TF*IDF score (weight), the rarer and more important the term, and vice versa.
#IDF - The specificity of a term can be quantified as an inverse function of the number of documents in which it occurs. 
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(ds['description'])


#Here we’ve calculated the cosine similarity of each item with every other item in the dataset, and then arranged them according to their similarity with item i, and stored the values in results
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix) 
results = {}
for idx, row in ds.iterrows():
   similar_indices = cosine_similarities[idx].argsort()[:-23:-1] 
   similar_items = [(cosine_similarities[idx][i], ds['product_id'][i]) for i in similar_indices] 
   results[row['product_id']] = similar_items[1:]

#Making Recommendation
def item(id):  
    return ds.loc[ds['product_id'] == id]['product_name'].tolist()[0].split(' - ')[0] 


# Just reads the results out of the dictionary.def 
def recommend(product_id):
    print("Recommending 20 products similar to " + item(product_id) + "...")   
    print("-------")   
    recs = results[product_id]  
    for rec in recs: 
        print("Recommended: " + item(rec[1]) + " (score:" +      str(rec[0]) + ")")


recommend(product_id=3566)


#Create data frame of similar products 
df = pd.DataFrame(results)
df = df.transpose()
new = pd.DataFrame(df.index)
new.rename(columns = {0 : 'product_id'},inplace = True)
new =  pd.merge(new, ke_skus[['product_id','product_name']], how = 'left', on = 'product_id')
counter = 1
for col in df.columns:
    a = pd.DataFrame(df[col].tolist(), index=df.index)
    a.reset_index(inplace = True)
    a = pd.merge(a,ke_skus[['product_id','product_name']], how = 'left' , left_on=1, right_on='product_id')
    a.drop(['product_id'], axis=1, inplace = True)
    a.rename(columns = {'index' : 'product_id', 0 : 'score' + str(counter), 1 : 'product' + str(counter), 'product_name' : 'product_name' + str(counter)}, inplace = True)
    new = pd.merge(new,a, how = 'left', on = 'product_id')
    counter += 1

new.to_csv('data/recommend.csv',index = False)






# Just reads the results out of the dictionary.def 
def recommend(item_id, num):
    print("Recommending " + str(num) + " products similar to " + item(item_id) + "...")   
    print("-------")   
    recs = results[item_id][:num]   
    for rec in recs: 
        print("Recommended: " + item(rec[1]) + " (score:" +      str(rec[0]) + ")")


recommend(item_id=3566, num=15)

