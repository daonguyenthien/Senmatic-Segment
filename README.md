# Senmatic-Segment (TO DO)
* NOTE : DÙNG TENSORFLOW VERSION 1.15

      tensorflow_version=1.15
      
Sau khi tải source về và tải lên drive kèm dataset bạn vào colab trỏ tới thư mực trong drive và tiến hành config 
- lệnh trỏ về drive  

      from google.colab import drive
      drive.mount('/content/drive')
      
- Chỉnh sửa các dòng trong file train.py để phù hợp với data
    - Dòng 32 'epoch_start_i' sửa default=0 cho lần chạy đầu tiên
    - Dòng 36 'continue_training' sửa default=False không tiếp tục train ( lần chạy đầu tiên )
    - Dòng 37 'dataset' sửa default=" tên folder của data " 
    - Dòng 38,39 chỉnh default về witdh và height của ảnh train
- Vào builders/model_builder.py comment hoặc xóa hết các module dòng 6,8->19 ( KHÔNG XÓA DÒNG SỐ 7 )
- Cuối cùng các bạn cd về folder chứa file train.py và chạy file train.py
    
    CHÚC CÁC BẠN THÀNH CÔNG ♥
     
