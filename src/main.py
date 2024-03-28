import sys
sys.path.append('C:/Users/ajboo/BookAbraham/KineticsProjects/imatinib_binding_analysis')


from kinetics_data_handling import dissociation_data_handling
from kinetics_data_handling import atp_data_handling
from kinetics_data_handling import imatinib_data_handling

def main():
    # Example 1: Equilibrium Binding of Imatinib to BCR-ABL1
    imatinib_file_path = "data/imatinib_binding_data.csv"
    imatinib_data = imatinib_data_handling.read_imatinib_binding_data(
        imatinib_file_path)
    cleaned_imatinib_data = imatinib_data_handling.clean_imatinib_binding_data(
        imatinib_data)
    normalized_imatinib_data = imatinib_data_handling.normalize_imatinib_binding_data(
        cleaned_imatinib_data)
    organized_imatinib_data = imatinib_data_handling.organize_imatinib_binding_data(
        normalized_imatinib_data)
    train_imatinib_data, test_imatinib_data = imatinib_data_handling.split_imatinib_binding_data(
        normalized_imatinib_data)

    # Example 2: ATP Equilibrium Binding
    atp_file_path = "data/ATP_equilibrium_data.csv"
    atp_data = atp_data_handling.read_atp_binding_data(atp_file_path)
    cleaned_atp_data = atp_data_handling.clean_atp_binding_data(atp_data)
    normalized_atp_data = atp_data_handling.normalize_atp_binding_data(
        cleaned_atp_data)
    organized_atp_data = atp_data_handling.organize_atp_binding_data(
        normalized_atp_data)
    train_atp_data, test_atp_data = atp_data_handling.split_atp_binding_data(
        normalized_atp_data)

    # Example 3: Dissociation Data
    dissociation_file_path = "data/dissociation_data.csv"
    dissociation_data = dissociation_data_handling.read_dissociation_data(
        dissociation_file_path)
    cleaned_dissociation_data = dissociation_data_handling.clean_dissociation_data(
        dissociation_data)
    organized_dissociation_data = dissociation_data_handling.organize_dissociation_data(
        cleaned_dissociation_data)
    train_dissociation_data, test_dissociation_data = dissociation_data_handling.split_dissociation_data(
        cleaned_dissociation_data)


if __name__ == "__main__":
    main()
