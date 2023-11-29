run:
	python src/solvmate/app/app.py 

open:
	open "http://127.0.0.1:8890"

all_fast:
	make clean 
	make training_data
	make pairs
	make train_recommender

all_slow:
	make clean 
	make download_data 
	make xtb_features 
	make training_data
	make pairs
	make train_recommender

clean:
	make clean_training_data 
	make clean_pairs  
	make clean_recommender 


download_data:
	python scripts/download_data.py

clean_pairs:
	rm "data/pairs.db" || echo "skipping clean pairs" 

clean_training_data:
	rm "data/training_data.db" || echo "skipping clean training data" 

clean_recommender:
	rm "data/recommender.pkl" || echo "skipping clean recommender"


pairs:
	python scripts/build_pairs.py

training_data:
	python scripts/build_training_data.py

xtb_features:
	python scripts/build_xtb_features.py

train_recommender:
	python scripts/train_recommender.py --finalretrain

hyperp_recommender:
	python scripts/train_recommender.py --hyperp

smi_to_name:
	python scripts/build_smi_to_name.py 

delete_solvent_selection_db:
	rm data/solvent_selection.db

cert_https:
	cd cert && openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365 -nodes

