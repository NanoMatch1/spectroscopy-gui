
def main():
    print("this is the test")
    print(app)
    ds.report_metadata()
    ds.remove_redundant_dat()
    ds.report_metadata()
    

def remove_redundant_dat() -> None:
    """Remove redundant 'dat' entries that already have an entry as ".acc" files"""
    self = ds # for interactive testing in the console
    print("Checking for redundant .dat entries...")
    redundant_keys = []
    for entry_key, entry in self._store.items():
        print(entry_key)
        series_id = entry.key.series_id
        if series_id.endswith(".dat"):
            print(f"Found .dat entry: {entry.key.key_tuple()}")
            data_key = entry.key
            key_tuple = data_key.key_tuple()
            acc_key_tuple = (key_tuple[0].replace(".dat", ".acc"), key_tuple[1], key_tuple[2])
            print(f"Checking for redundant entry: {key_tuple} → looking for {acc_key_tuple}")
            if acc_key_tuple in self._store:
                redundant_keys.append(entry_key)

    keys = sorted(redundant_keys)
    print('---- L ist of redundant .dat entries to remove ----')
    for key_tuple in keys:
        print(key_tuple)
        # # print(f"Metadata: {entry.metadata}")
        # if entry.metadata.get("data_type") == "dat":
            
        #     print(f"Found .dat entry: {entry.key.key_tuple()}")
        # else:
        #     print(f"Not a .dat entry: {entry.key.key_tuple()}")
    #         data_key = entry.key
    #         key_tuple = data_key.key_tuple()
    #         acc_key_tuple = (key_tuple[0].replace(".dat", ".acc"), key_tuple[1], key_tuple[2])
    #         print(f"Checking for redundant entry: {key_tuple} → looking for {acc_key_tuple}")
    #         if acc_key_tuple in self._store:
    #             redundant_keys.append(key_tuple)
    #     elif entry.metadata.get("file_type") == "acc":
    #         print(f"Found .acc entry: {entry.key.key_tuple()}")
    # for key_tuple in redundant_keys:
    #     self.remove(DataKey(*key_tuple))
    #     print(f"Removed redundant entry: {key_tuple}")
# main()

ret = remove_redundant_dat()

