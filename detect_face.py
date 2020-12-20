import face_extractor
import Codebook as cb

#extractor = face_extractor.FaceExtractor('faces', 'cropped_images')
#extractor.preprocess_images()

codebook = cb.Codebook()
#codebook.create_codebook()
v_angle, h_angle = codebook.Estimate_angles_for_img('cropped_images_copy/Person15/0person15115-30-75.jpg', 'Codebook_cell8x8')
print("Estimated orientation: Vertical= {}, Horizontal= {}".format(v_angle, h_angle))