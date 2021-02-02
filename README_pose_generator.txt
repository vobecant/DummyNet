Download data
1. wget https://data.ciirc.cvut.cz/public/projects/DummyNet/joints_pca_etc.npz
2. wget https://data.ciirc.cvut.cz/public/projects/DummyNet/pca_per_cluster.zip
3. extract "pca_per_cluster.zip" to "pca_per_cluster" folder next to joints_pca_etc.npz
4. set opt['load_path'] to point into the folder, where you have joints_pca_etc.npy and pca_per_cluster folder
5. set opt['save_path'] to a folder where generated poses should be stored

Dependencies:
numpy 1.16.5
matplotlib 3.1.1
jsonschema 3.0.2
sklearn 0.21.2 (0.21.3 generates warning, but works too)
joblib 0.13.2

Run:
python pose_generator.py