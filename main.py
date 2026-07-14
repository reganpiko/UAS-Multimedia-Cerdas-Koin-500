import cv2
from ultralytics import YOLO

# 1. Muat model
model = YOLO('best.pt') 

# 2. Buka webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Kamera tidak terdeteksi/terbaca.")
        break

    # 3. Deteksi koin
    results = model(frame)
    jumlah_koin_500 = 0
    
    # 4. Logika memisahkan koin 500
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            
            # Cek apakah nama objeknya mengandung kata "500"
            # (Berguna jika nama kelas di datasetmu seperti "koin_500", "500", atau "Rp500")
            if "500" in class_name:
                jumlah_koin_500 += 1
                
                # Ambil titik koordinat sudut kotak koin tersebut
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Gambar kotak (warna hijau) HANYA untuk koin 500
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Tambahkan teks label kecil di atas kotaknya
                cv2.putText(frame, "Koin 500", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 5. Tulis teks jumlah koin di pojok kiri atas
    teks = f'Total Koin 500: {jumlah_koin_500}'
    cv2.putText(frame, teks, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # 6. Tampilkan ke layar
    cv2.imshow("Deteksi Koin 500 Perak - Mini Expo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()