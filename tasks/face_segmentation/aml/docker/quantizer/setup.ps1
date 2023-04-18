# assumes you have already run "az login" and "az account set --subscription id"
# also assumes you have run "az aks install-cli"

$resource_group = "snpe-quantizaton-rg"
$storage_account_name = "nasfacemodels"
$plan_location = "westus2"
$aks_cluster = "snpe-quantizer-aks"
$aks_node_vm = "Standard_D16s_v3"
$aks_namespace = "snpe"

# This registry name is also  referenced in quantizer.yaml
$registry_name = "snpecontainerregistry001"

function Check-Provisioning($result) {
    $rc = Join-String -Separator "`n" -InputObject $rc
    $x = ConvertFrom-Json $rc
    if ($x.provisioningState -ne "Succeeded") {
        Write-Error "Failed to create registry"
        Write-Error $rc
        exit 1
    }
}

function GetConnectionString()
{
    $x = az storage account show-connection-string --name $storage_account_name  --resource-group $resource_group | ConvertFrom-Json
    return $x.connectionString
}

function CreateBlobContainer($name, $conn_str)
{
    Write-Host "Checking blob container '$name' exists"
    $rc = &az storage container exists --name $name --connection-string $conn_str  | ConvertFrom-Json
    if (-not $rc.exists) {
        Write-Host "Creating blob container $name"
        $rc = &az storage container create --name $name --resource-group $resource_group --connection-string $conn_str  | ConvertFrom-Json
        if (-not $rc.created) {
            Write-Error "Failed to create blob container $name"
            Write-Error $rc
        }
    }
}

function EnablePublicAccess($name, $conn_str){
    Write-Host "Checking blob container '$name' has public access"
    $rc = &az storage container show-permission --name $name --connection-string $conn_str | ConvertFrom-Json
    if ($rc.publicAccess -ne "blob" ){
        Write-Host "Setting blob container '$name' permissions for public access"
        $rc = &az storage container set-permission --name $name --public-access blob --connection-string $conn_str
    }
}

function GetZipRootName($zip)
{
    $zip = [IO.Compression.ZipFile]::OpenRead($zip)
    $entries = $zip.Entries
    $root = [System.IO.Path]::GetDirectoryName($entries[0].FullName)
    $zip.Dispose()
    return $root
}

function CopyLocal($zip)
{
    $a = [System.IO.Path]::GetDirectoryName([System.IO.Path]::GetFullPath($zip))
    $b = [System.IO.Path]::GetFullPath(".")
    if ($a -ne $b) {
        Copy-Item -Path $zip -Destination "."
    }
}

function EnsureNamespace($name)
{
    Write-Host "Checking kubernetes namespaces"
    $rc = &kubectl get namespace $name 2>&1
    if ($rc.ToString().Contains("NotFound")) {
        Write-Host "Creating kubernetes namespace $name" 2>&1
        $rc = &kubectl create namespace $name
        Write-Host "Create returned: $rc"
    }
}

if ("$Env:SNPE_SDK" -eq "")
{
    Write-Host "Please set your SNPE_SDK path so we can upload the SNPE SDK zip file to Azure"
    exit 1
}

CopyLocal -zip $Env:SNPE_SDK
$snpe_sdk_zip = [System.IO.Path]::GetFileName($Env:SNPE_SDK)
$snpe_root = GetZipRootName -zip $Env:SNPE_SDK

if ("$Env:ANDROID_NDK" -eq ""){
    Write-Host "Please set your ANDROID_NDK path so we can upload the Android NDK zip file to Azure"
    exit 1
}

CopyLocal -zip $Env:ANDROID_NDK
$android_sdk_zip = [System.IO.Path]::GetFileName($Env:ANDROID_NDK)
$android_ndk_root = GetZipRootName -zip $Env:ANDROID_NDK

if ("$Env:INPUT_TESTSET" -eq ""){
    Write-Host "Please set your INPUT_TESTSET path pointing to zip file containing the 10k test set images"
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

CreateBlobContainer -name "models" -conn_str $conn_str
CreateBlobContainer -name "downloads" -conn_str $conn_str
EnablePublicAccess -name "downloads" -conn_str $conn_str

Write-Host Checking container registry $registry_name
$rc = &az acr show --resource-group $resource_group --name $registry_name 2>&1
if ($rc.Contains("ResourceNotFound")) {
    Write-Host Creating container registry $registry_name...
    $rc = &az acr create --resource-group $resource_group --name $registry_name --sku Standard 2>&1
    Check-Provisioning -result $rc
    $rc = &az acr update --name $registry_name --anonymous-pull-enabled true --admin-enabled true
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

$rc = &"$env:windir\system32\where" kubectl.exe 2>&1
print("where kubectl.exe => $rc")
if ($rc.ToString().Contains("Could not find files")) {
    Write-Host "kubectl not found, skipping kubectl setup."
    if ($IsWindows){
        Write-Host "You can build the quantizer docker image on Windows if you install the Docker Desktop for Windows"
        Write-Host "and set it to Linux container mode and you can manage your Azure Kubernetes cluster using kubectl if"
        Write-Host "you enable the docker desktop Kubernetes support under Settings."
    }
}
else
{
    # this makes it possible to use kubectl locally to manage this cluster.
    &az aks get-credentials --resource-group $resource_group --name $aks_cluster
    EnsureNamespace -name $aks_namespace
}

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

# populate MODEL_STORAGE_CONNECTION_STRING variable in quantizer.template.yaml
$template = Get-Content -Path "quantizer.template.yaml"

$tags = &az acr repository show-tags -n $registry_name --repository quantizer | ConvertFrom-JSON

if ($tags.GetType().Name -eq "String"){
    $tags = @($tags)
}

$latest = [Version]"1.0"
foreach ($t in $tags) {
    $v = [Version]$t
    if ($v -gt $latest){
        $latest = $v
    }
}

$v = [Version]$latest
$vnext = [System.String]::Format("{0}.{1}", $v.Major, $v.Minor + 1)
Write-Host "Creating quantizer.yaml and setting image version $vnext"
$template = $template.Replace("quantizer:1.0", "quantizer:$vnext")
$template = $template.Replace("$MSCS$", $conn_str)
Set-Content -Path "quantizer.yaml" -Value $template

# ================= Write out info/next steps ================

Write-Host ""
Write-Host docker build `
  --build-arg "SNPE_SDK_ZIP=$snpe_sdk_zip" --build-arg "SNPE_SDK_ROOT=$snpe_root" `
  --build-arg "ANDROID_NDK_ZIP=$android_sdk_zip" --build-arg "ANDROID_NDK_ROOT=$android_ndk_root" `
  . --progress plain

Write-Host ""
Write-Host "### Run the above docker build to build the docker image and the following to publish it..."
Write-Host "Get the password from Azure portal for $registry_name under Access keys:"
Write-Host "  docker login $acrServer -u $registry_name -p <get password>"
Write-Host "  az aks get-credentials --resource-group $resource_group --name $aks_cluster"
Write-Host "  docker tag ... $registry_name.azurecr.io/quantizer:$vnext"
Write-Host "  docker push $registry_name.azurecr.io/quantizer:$vnext"
Write-Host ""
Write-Host "To cleanup old images see cleanup.ps1"
Write-Host ""
Write-Host "### Apply the new image on your cluster..."
Write-Host "  kubectl apply -f quantizer.yaml"

Write-Host ""
Write-Host "### To run the runner script locally please set the following environment variable: "
Write-HOst "set MODEL_STORAGE_CONNECTION_STRING=$conn_str"