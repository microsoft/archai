# assumes you have already run "az login" and "az account set --subscription id"
# also assume you have run "az aks install-cli"

$resource_group = "snpe-quantizaton-rg"
$registry_name = "snpecontainerregistry001"
$storage_account_name = "nasmodelstorage"
$plan_location = "westus2"
$aks_cluster = "snpe-quantizer-aks"
$aks_node_vm = "Standard_D16s_v3"
$storage_container = "nasfacemodels"

function Check-Provisioning($result) {
    $rc = Join-String -Separator "`n" -InputObject $rc
    $x = ConvertFrom-Json $rc
    if ($x.provisioningState -ne "Succeeded") {
        Write-Host "Failed to create registry"
        Write-Host $rc
        exit 1
    }
}

function GetConnectionString()
{
    $x = az storage account show-connection-string --name $storage_account_name  --resource-group $resource_group | ConvertFrom-Json
    return $x.connectionString
}

function GetZipRootName($zip)
{
    $zip = [IO.Compression.ZipFile]::OpenRead($zip)
    $entries = $zip.Entries
    $root = [System.IO.Path]::GetDirectoryName($entries[0].FullName)
    $zip.Dispose()
    return $root
}

if ("$Env:SNPE_SDK" -eq "")
{
    Write-Host "Please set your SNPE_SDK path so we can upload the SNPE SDK zip file to Azure"
    exit 1
}

if ("$Env:ANDROID_NDK" -eq ""){
    Write-Host "Please set your ANDROID_NDK path so we can upload the Android NDK zip file to Azure"
    exit 1
}

if ("$Env:INPUT_TESTSET" -eq ""){
    Write-Host "Please set your INPUT_TESTSET path pointing to zip file containing the 10k test set images"
    exit 1
}

if ("$Env:GITHUB_PAT" -eq ""){
    Write-Host "Please set your GITHUB_PAT so we can access the https://github.com/lovettchris/snpe_runner repository"
    exit 1
}

if ("$Env:MODEL_STORAGE_CONNECTION_STRING" -eq ""){
    Write-Host "Please set your MODEL_STORAGE_CONNECTION_STRING to specify which storage account to use for the model runner."
    exit 1
}

if (-not $Env:MODEL_STORAGE_CONNECTION_STRING.contains($storage_container)){
    Write-Host "Please check your MODEL_STORAGE_CONNECTION_STRING is pointing to $storage_container."
    exit 1
}

Write-Host "Checking azure account..."
$output = &az account show 2>&1
if ($output.Contains("ERROR:"))
{
    Write-Host "Please login to an azure account using 'az login' and set the subscription you want to use."
    Write-Host "using 'az account set --subscriptoin id' then try again."
    Exit 1
}

$account = $output | ConvertFrom-Json
$name = $account.name
$subscription = $account.id
Write-Host "You are using azure subscription $name with id $subscription"
$proceed = Read-Host -Prompt "Do you want to proceed (y/n) "
$lower = $proceed.ToLowerInvariant()
if ($lower -ne 'yes' -and $lower -ne 'y'){
    Write-Host "Your answer $lower does not match 'yes' or 'y' so the script is terminating."
    Exit 1
}


# ======= Storage account
Write-Host Checking storage account $storage_account_name...
$rc = &az storage account show --resource-group $resource_group --name $storage_account_name 2>&1
$rc = Join-String -Separator "`n" -InputObject $rc
if ($rc.Contains("ResourceNotFound")) {
    Write-Host Creating storage account $storage_account_name...
    $rc = &az storage account create --name $storage_account_name --resource-group $resource_group --location $plan_location --kind StorageV2 --sku Standard_LRS 2>&1
}
Check-Provisioning -result $rc

$conn_str = GetConnectionString
Write-Host "Connection string is $conn_str"

Write-Host Checking container registry $registry_name
$rc = &az acr show --resource-group $resource_group --name $registry_name 2>&1
if ($rc.Contains("ResourceNotFound")) {
    Write-Host Creating container registry $registry_name...
    $rc = &az acr create --resource-group $resource_group --name $registry_name --sku Standard 2>&1
    Check-Provisioning -result $rc
    $rc = &az acr az acr update --name $resource_group --anonymous-pull-enabled true --admin-enabled true 
}

Check-Provisioning -result $rc

$rc = &az acr login --name $registry_name --expose-token | ConvertFrom-Json
$token = $rc.accessToken
$acrServer = $rc.loginServer

# ======= aks cluster
Write-Host Checking aks cluster..
$rc = &az aks show --name $aks_cluster --resource-group $resource_group 2>&1
if ($rc.Contains("ResourceNotFound")) {
    # note, azure requires min of 10 pods per node, but our pods are running expensive quantization job, so
    # to allow 10 of those to run on a node we've had to scale up our VM to Standard_D16s_v3 with 16 cores and 64 gb RAM.
    # even then we'll see how well that performs... could be 10 quantizations on a node will thrash it to bits...
    $rc = &az aks create --name $aks_cluster --resource-group $resource_group --location $plan_location --enable-cluster-autoscaler `
              --node-osdisk-size 250 --min-count 1 --max-count 10 --max-pods 10 --node-osdisk-size 100 --node-vm-size $aks_node_vm `
              --node-osdisk-type Managed --nodepool-name nodepool1   2>&1
}

Check-Provisioning -result $rc

Write-Host " ======= upload SNPE_SDK"
$fileName = [System.IO.Path]::GetFileName($Env:SNPE_SDK)
$rc = az storage blob exists --account-name $storage_account_name --container-name downloads --name $fileName --connection-string  $conn_str  | ConvertFrom-Json
if ($rc.exists){
    Write-Host "File $fileName already exists in 'downloads' container"
} else {
    Write-Host "Uploading $fileName to 'downloads' container"
    az storage blob upload --file "$Env:SNPE_SDK" --account-name $storage_account_name --container-name downloads --name $fileName --connection-string "$conn_str"
}
$snpe_sdk_url="https://$storage_account_name.blob.core.windows.net/downloads/$fileName"
$snpe_sdk_filename=$fileName
$snpe_root = GetZipRootName -zip $Env:SNPE_SDK


Write-Host "======= upload ANDROID_NDK "
$fileName = [System.IO.Path]::GetFileName($Env:ANDROID_NDK)
$rc = az storage blob exists --account-name $storage_account_name --container-name downloads --name $fileName --connection-string  $conn_str  | ConvertFrom-Json
if ($rc.exists){
    Write-Host "File $fileName already exists in 'downloads' container"
} else {
    Write-Host "Uploading $fileName to 'downloads' container"
    az storage blob upload --file "$Env:ANDROID_NDK" --account-name $storage_account_name --container-name downloads --name $fileName --connection-string "$conn_str"
}

$android_ndk_url="https://$storage_account_name.blob.core.windows.net/downloads/$fileName"
$android_ndk_filename=$fileName
$android_ndk_root = GetZipRootName -zip $Env:ANDROID_NDK

Write-Host "======= upload INPUT_TESTSET"
$fileName = [System.IO.Path]::GetFileName($Env:INPUT_TESTSET)
$rc = az storage blob exists --account-name $storage_account_name --container-name downloads --name $fileName --connection-string  $conn_str  | ConvertFrom-Json
if ($rc.exists){
    Write-Host "File $fileName already exists in 'downloads' container"
} else {
    Write-Host "Uploading $fileName to 'downloads' container"
    az storage blob upload --file "$Env:INPUT_TESTSET" --account-name $storage_account_name --container-name downloads --name $fileName --connection-string "$conn_str"
}

$test_set_url="https://$storage_account_name.blob.core.windows.net/downloads/$fileName"
Write-Host "Test set url is $test_set_url"

# ================= Write out info/next steps ================
Write-Host docker build --build-arg "`"MODEL_STORAGE_CONNECTION_STRING=$Env:MODEL_STORAGE_CONNECTION_STRING`"" `
  --build-arg "SNPE_SDK=$snpe_sdk_url" --build-arg "SNPE_SDK_FILENAME=$snpe_sdk_filename" --build-arg "SNPE_ROOT=/home/chris/$snpe_root" `
  --build-arg "ANDROID_NDK=$android_ndk_url" --build-arg "ANDROID_NDK_FILENAME=$android_ndk_filename" --build-arg "ANDROID_NDK_ROOT=/home/chris/$android_ndk_root" `
  --build-arg "GITHUB_PAT=$Env:GITHUB_PAT" . --progress plain

Write-Host ""
Write-Host "### Run the above docker builld to build the docker image and the following to publish it..."
Write-Host "Get the password from Azure portal for $registry_name under Access keys:"
Write-Host "  docker login $acrServer -u $registry_name -p <get password>"
Write-Host "  docker tag ... snpecontainerregistry001.azurecr.io/quantizer:1.1"
Write-Host "  docker push snpecontainerregistry001.azurecr.io/quantizer:1.1"
Write-Host "  az acr repository delete --name $registry_name --image <previous_version> -u $registry_name -p <get password>"
Write-Host "### Make sure the right image version is mentioned in quant.yaml..."
Write-Host "  az aks get-credentials --resource-group $resource_group --name $aks_cluster"
Write-Host "  kubectl apply -f quant.yaml"

# $output = Join-String -Separator "`n" -InputObject $rc
# Write-Host $output
