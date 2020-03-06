from keras.models import load_model
import functions

model =load_model('face-rec_Google.h5')
functions.main(model)